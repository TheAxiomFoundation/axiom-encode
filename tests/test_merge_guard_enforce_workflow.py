"""Tripwires for the enforcing merge guard: its workflow, bundle and launcher.

``.github/workflows/content-bound-merge-guard.yml`` is the workflow an
organization ruleset would require on RuleSpec repositories (change 3b of
``docs/content-bound-merge-guard.md`` section 8). No ruleset exists; the
workflow is required nowhere. This file pins:

- the workflow: exact triggers with no filter, exact read permissions, one
  job with no skip path, no secret, variable, environment, ``id-token`` or
  concurrency, pinned ``actions/`` steps, the sparse checkout of exactly the
  bundle, ``python3 -I -S``, and the bundle digest literal;
- the bundle: each file imports only the standard library and listed bundle
  modules, at every depth of the file;
- the launcher: under ``-I -S`` a sweep over every lane loads no module
  outside the list, and the launcher refuses every other way to start it.

It is a tripwire, not the control. A pull request can edit it together with
the workflow; ``CODEOWNERS`` owns both. The control is the organization
ruleset's pin of the workflow file to a commit.

Each rule has a negative case that changes the workflow (or a bundle file)
in memory and requires the tripwire to report it.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from axiom_encode import content_merge_guard as guard
from axiom_encode import merge_guard_runner as runner
from tests.test_content_merge_guard import (
    CHECKS,
    REPO,
    ROOT,
    H,
    build,
    plain,
    queue,
)
from tests.test_merge_guard_runner import (
    TOKEN,
    V5_LANE,
    group_event,
    pr_event,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = REPOSITORY_ROOT / ".github" / "workflows"
WORKFLOW_NAME = "content-bound-merge-guard.yml"
WORKFLOW_PATH = WORKFLOWS / WORKFLOW_NAME
LAUNCHER_PATH = REPOSITORY_ROOT / "scripts" / "merge_guard_launch.py"
DRIVER_PATH = REPOSITORY_ROOT / "tests" / "merge_guard_bundle_driver.py"

# The actions the job may call, at the commits this repository already pins.
CHECKOUT = "actions/checkout@de0fac2e4500dabe0009e67214ff5f5447ce83dd"
SETUP_PYTHON = "actions/setup-python@a309ff8b426b58ec0e2a45f0f869d46889d02405"
PERMISSIONS = {
    "contents": "read",
    "pull-requests": "read",
    "checks": "read",
    "actions": "read",
}
EVALUATE_SCRIPT = 'set -euo pipefail\npython3 -I -S "$RUNNER_TEMP/guard/launch.py"\n'
WORKFLOW_SHA = "${{ github.workflow_sha }}"
JOB_TOKEN = "${{ github.token }}"


def _load_launcher():
    spec = importlib.util.spec_from_file_location("merge_guard_launch", LAUNCHER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


launch = _load_launcher()


def workflow_text() -> str:
    return WORKFLOW_PATH.read_text(encoding="utf-8")


def workflow() -> dict:
    return yaml.safe_load(workflow_text())


def other_workflows() -> dict[str, object]:
    paths = sorted([*WORKFLOWS.glob("*.yml"), *WORKFLOWS.glob("*.yaml")])
    return {
        path.name: yaml.safe_load(path.read_text(encoding="utf-8"))
        for path in paths
        if path.name != WORKFLOW_NAME
    }


# --- the bundle digest ---------------------------------------------------------
#
# Format: for each path of ``launch.BUNDLE_FILES``, in that order, one line
# ``<sha256 hex of the file><two spaces><path>\n``; the digest is the sha256
# of those lines. It is what
# ``sha256sum <paths in that order> | sha256sum`` prints, run in the bundle
# directory. ``axiom_encode/__init__.py`` is the empty file.


def bundle_sources(root: Path = REPOSITORY_ROOT) -> dict[str, bytes]:
    """Bundle path -> bytes, from the working tree. No Git history is read."""
    sources = {
        launch.LAUNCHER_NAME: (root / launch.LAUNCHER_SOURCE).read_bytes(),
        f"{launch.PACKAGE}/__init__.py": b"",
    }
    for module in launch.BUNDLE_MODULES:
        sources[f"{launch.PACKAGE}/{module}.py"] = (
            root / "src" / launch.PACKAGE / f"{module}.py"
        ).read_bytes()
    return sources


def bundle_digest(sources: dict[str, bytes]) -> str:
    assert tuple(sources) == launch.BUNDLE_FILES
    lines = "".join(
        f"{hashlib.sha256(sources[path]).hexdigest()}  {path}\n"
        for path in launch.BUNDLE_FILES
    )
    return hashlib.sha256(lines.encode("ascii")).hexdigest()


# --- the tripwire ----------------------------------------------------------------


def _triggers(document: dict) -> object:
    # PyYAML reads the bare key ``on`` as the boolean True.
    return document.get("on", document.get(True))


def _strings(value: object):
    """Every string scalar of a parsed YAML value, keys included."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from _strings(key)
            yield from _strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)


def _keys(value: object):
    if isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from _keys(item)
    elif isinstance(value, list):
        for item in value:
            yield from _keys(item)


FORBIDDEN_KEYS = frozenset(
    {
        "if",
        "continue-on-error",
        "environment",
        "concurrency",
        "secrets",
        "id-token",
        "needs",
        "strategy",
        "container",
        "services",
        "defaults",
        "shell",
        "working-directory",
        "cache",
        "token",
    }
)
FORBIDDEN_TEXT = (
    "secrets.",
    "secrets[",
    "vars.",
    "vars[",
    "inputs.",
    "id-token",
    "cancel-in-progress",
    "pull_request_target",
    "workflow_call",
    "workflow_dispatch",
    "uv ",
    "pip ",
    "curl ",
    "wget ",
    "| tee",
)
_EXPRESSION = re.compile(r"\$\{\{.*?\}\}", re.S)


def could_publish(name: object, check_name: str) -> bool:
    """Whether a job or workflow name can evaluate to ``check_name``.

    An expression in the name can evaluate to any text, so it matches
    anything in its place.
    """
    if not isinstance(name, str):
        return False
    literals = [re.escape(part.strip()) for part in _EXPRESSION.split(name)]
    pattern = ".*".join(literals) if len(literals) > 1 else literals[0]
    return (
        re.fullmatch(pattern, check_name, re.I | re.S) is not None
        or name.strip().casefold() == check_name.casefold()
    )


def workflow_problems(
    text: str, others: dict[str, object], digest: str, *, wait_seconds: int
) -> list[str]:
    """Every way the workflow text departs from the reviewed shape."""
    try:
        document = yaml.safe_load(text)
    except yaml.YAMLError:
        return ["workflow: not YAML"]
    if not isinstance(document, dict):
        return ["workflow: not a mapping"]
    problems: list[str] = []
    name = runner.GUARD_CHECK_NAME

    if set(document) != {"name", True, "permissions", "jobs"}:
        problems.append("workflow: top-level keys are not name, on, permissions, jobs")
    if document.get("name") != name:
        problems.append("workflow: name is not the guard name")
    triggers = _triggers(document)
    if not isinstance(triggers, dict) or set(triggers) != {
        "pull_request",
        "merge_group",
    }:
        problems.append("triggers: not exactly pull_request and merge_group")
    elif any(value is not None for value in triggers.values()):
        problems.append("triggers: a trigger has a filter or activity types")
    if document.get("permissions") != PERMISSIONS:
        problems.append("permissions: not exactly the four read permissions")

    for key in sorted({str(key) for key in _keys(document)} & FORBIDDEN_KEYS):
        problems.append(f"forbidden key: {key}")
    strings = list(_strings(document))
    for needle in FORBIDDEN_TEXT:
        if any(needle in value for value in strings):
            problems.append(f"forbidden text: {needle}")
    expressions = sorted(
        {match for value in strings for match in _EXPRESSION.findall(value)}
        | {value for value in strings if "${{" in value and "}}" not in value}
    )
    if expressions != sorted([WORKFLOW_SHA, JOB_TOKEN]):
        problems.append("expressions: not exactly the workflow SHA and the job token")

    jobs = document.get("jobs")
    if not isinstance(jobs, dict) or list(jobs) != ["guard"]:
        problems.append("jobs: not exactly one job, guard")
        return problems
    job = jobs["guard"]
    if not isinstance(job, dict):
        problems.append("job: not a mapping")
        return problems
    if set(job) != {"name", "runs-on", "timeout-minutes", "steps"}:
        problems.append("job: keys are not name, runs-on, timeout-minutes, steps")
    if job.get("name") != name:
        problems.append("job: name is not the guard check name")
    if job.get("runs-on") != "ubuntu-latest":
        problems.append("job: runs-on is not ubuntu-latest")
    timeout = job.get("timeout-minutes")
    if type(timeout) is not int or timeout * 60 < wait_seconds + 600:
        problems.append("job: timeout does not cover the CI wait and a margin")

    steps = job.get("steps")
    if (
        not isinstance(steps, list)
        or len(steps) != 4
        or not all(isinstance(step, dict) for step in steps)
    ):
        problems.append("steps: not exactly four steps")
        return problems
    checkout, python, build_step, evaluate = steps
    for step in steps:
        uses = step.get("uses")
        if "uses" in step and not (
            isinstance(uses, str)
            and re.fullmatch(r"actions/[a-z-]+@[0-9a-f]{40}", uses)
        ):
            problems.append("uses: a step is not an actions/ action at a full SHA")
        if "run" in step:
            run = step["run"]
            if not isinstance(run, str) or not run.startswith("set -euo pipefail\n"):
                problems.append("run: a script does not start with set -euo pipefail")
            if not isinstance(run, str) or "${{" in run:
                problems.append("run: a script holds a workflow expression")

    if set(checkout) != {"name", "uses", "with"} or checkout.get("uses") != CHECKOUT:
        problems.append("checkout: not the pinned checkout step")
    options = checkout.get("with")
    options = options if isinstance(options, dict) else {}
    if set(options) != {
        "repository",
        "ref",
        "path",
        "persist-credentials",
        "sparse-checkout",
        "sparse-checkout-cone-mode",
    }:
        problems.append("checkout: inputs are not the reviewed set")
    if options.get("repository") != guard.ENCODER_REPOSITORY:
        problems.append("checkout: repository is not axiom-encode")
    if options.get("ref") != WORKFLOW_SHA:
        problems.append("checkout: ref is not github.workflow_sha")
    if options.get("path") != "guard-source":
        problems.append("checkout: path is not guard-source")
    if options.get("persist-credentials") is not False:
        problems.append("checkout: persist-credentials is not false")
    if options.get("sparse-checkout-cone-mode") is not False:
        problems.append("checkout: sparse-checkout-cone-mode is not false")
    sparse = options.get("sparse-checkout")
    if not isinstance(sparse, str) or tuple(sparse.split()) != launch.SOURCE_FILES:
        problems.append("checkout: sparse-checkout is not the bundle list")

    if (
        set(python) != {"name", "uses", "with"}
        or python.get("uses") != SETUP_PYTHON
        or python.get("with") != {"python-version": "3.13"}
    ):
        problems.append("python: not the pinned setup-python 3.13 step with no cache")

    if set(build_step) != {"name", "env", "run"}:
        problems.append("build: keys are not name, env, run")
    environment = build_step.get("env")
    literal = (
        environment.get("BUNDLE_SHA256") if isinstance(environment, dict) else None
    )
    if (
        not isinstance(environment, dict)
        or set(environment) != {"BUNDLE_SHA256"}
        or not isinstance(literal, str)
        or re.fullmatch(r"[0-9a-f]{64}", literal) is None
    ):
        problems.append("build: env is not exactly one BUNDLE_SHA256 literal")
    elif literal != digest:
        problems.append(
            f"build: BUNDLE_SHA256 is {literal}, the bundle digest is {digest}"
        )

    if set(evaluate) != {"name", "env", "run"}:
        problems.append("evaluate: keys are not name, env, run")
    if evaluate.get("env") != {runner.TOKEN_ENV: JOB_TOKEN}:
        problems.append("evaluate: env is not exactly the job token")
    if evaluate.get("run") != EVALUATE_SCRIPT:
        problems.append("evaluate: script is not exactly python3 -I -S launch.py")

    stem = WORKFLOW_NAME.rsplit(".", 1)[0]
    for other_name, other in sorted(others.items()):
        if other_name.rsplit(".", 1)[0] == stem:
            problems.append(f"{other_name}: a second file with the guard's name")
        if not isinstance(other, dict):
            continue
        if could_publish(other.get("name"), name):
            problems.append(f"{other_name}: workflow name can be the guard name")
        other_jobs = other.get("jobs")
        for job_id, other_job in (
            other_jobs.items() if isinstance(other_jobs, dict) else ()
        ):
            where = f"{other_name}:{job_id}"
            if could_publish(job_id, name):
                problems.append(f"{where}: job id can publish the guard check name")
            if not isinstance(other_job, dict):
                continue
            if could_publish(other_job.get("name"), name):
                problems.append(f"{where}: job name can publish the guard check name")
            if stem in str(other_job.get("uses", "")):
                problems.append(f"{where}: calls the guard workflow")
            for step in other_job.get("steps") or ():
                if isinstance(step, dict) and stem + ".y" in str(step.get("uses", "")):
                    problems.append(f"{where}: a step uses the guard workflow")
    return problems


def problems_of(text: str, others: dict | None = None, digest: str | None = None):
    return workflow_problems(
        text,
        other_workflows() if others is None else others,
        bundle_digest(bundle_sources()) if digest is None else digest,
        wait_seconds=runner.CI_WAIT_SECONDS,
    )


def test_the_workflow_has_the_reviewed_shape():
    assert problems_of(workflow_text()) == []


def test_the_digest_literal_is_the_digest_of_the_working_tree_bundle():
    document = workflow()
    literal = document["jobs"]["guard"]["steps"][2]["env"]["BUNDLE_SHA256"]
    expected = bundle_digest(bundle_sources())
    assert literal == expected, (
        f"the bundle changed: set BUNDLE_SHA256 in {WORKFLOW_NAME} to {expected} "
        "in the same reviewed change"
    )


def test_the_job_publishes_the_reserved_check_name_and_covers_the_wait():
    job = workflow()["jobs"]["guard"]
    assert job["name"] == runner.GUARD_CHECK_NAME == "content-bound-merge-guard"
    assert workflow()["name"] == runner.GUARD_CHECK_NAME
    assert job["timeout-minutes"] == 75
    assert job["timeout-minutes"] * 60 >= runner.CI_WAIT_SECONDS + 600


def test_the_header_states_what_is_off_and_what_is_unverified():
    comments = " ".join(
        line.lstrip("# ").strip()
        for line in workflow_text().splitlines()
        if line.startswith("#")
    )
    for phrase in (
        "It is required nowhere.",
        "until an organization owner creates an organization ruleset",
        "No variable, key or repository setting switches it on.",
        '{"inert":"source_repository"}',
        "No live ruleset run has shown what `github.workflow_sha` is",
        "its digest differs from the BUNDLE_SHA256 literal of this file and the job fails",
        "a skipped job counts as passed",
    ):
        assert phrase in comments, phrase


def edited(change) -> str:
    document = workflow()
    change(document)
    return yaml.safe_dump(document, sort_keys=False)


def job_of(document):
    return document["jobs"]["guard"]


def step_of(document, index):
    return job_of(document)["steps"][index]


def set_in(target, key, value):
    def change(document):
        target(document)[key] = value

    return change


def set_trigger(value):
    def change(document):
        document[True] = value

    return change


NEGATIVE_CASES = {
    "trigger pull_request_target added": (
        set_trigger(
            {"pull_request": None, "merge_group": None, "pull_request_target": None}
        ),
        "triggers: not exactly",
    ),
    "trigger workflow_dispatch added": (
        set_trigger(
            {"pull_request": None, "merge_group": None, "workflow_dispatch": None}
        ),
        "triggers: not exactly",
    ),
    "trigger workflow_call added": (
        set_trigger({"pull_request": None, "merge_group": None, "workflow_call": None}),
        "triggers: not exactly",
    ),
    "trigger schedule added": (
        set_trigger(
            {
                "pull_request": None,
                "merge_group": None,
                "schedule": [{"cron": "0 * * * *"}],
            }
        ),
        "triggers: not exactly",
    ),
    "trigger merge_group dropped": (
        set_trigger({"pull_request": None}),
        "triggers: not exactly",
    ),
    "trigger pull_request dropped": (
        set_trigger({"merge_group": None}),
        "triggers: not exactly",
    ),
    "triggers as a list": (
        set_trigger(["pull_request", "merge_group"]),
        "triggers: not exactly",
    ),
    "trigger replaced by pull_request_target": (
        set_trigger({"pull_request_target": None, "merge_group": None}),
        "triggers: not exactly",
    ),
    "branch filter": (
        set_trigger({"pull_request": {"branches": ["main"]}, "merge_group": None}),
        "triggers: a trigger has a filter",
    ),
    "path filter": (
        set_trigger(
            {"pull_request": {"paths-ignore": ["docs/**"]}, "merge_group": None}
        ),
        "triggers: a trigger has a filter",
    ),
    "activity types": (
        set_trigger({"pull_request": {"types": ["opened"]}, "merge_group": None}),
        "triggers: a trigger has a filter",
    ),
    "merge_group types": (
        set_trigger(
            {"pull_request": None, "merge_group": {"types": ["checks_requested"]}}
        ),
        "triggers: a trigger has a filter",
    ),
    "job if": (set_in(job_of, "if", "github.repository != 'x'"), "forbidden key: if"),
    "step if": (
        set_in(lambda d: step_of(d, 3), "if", "always()"),
        "forbidden key: if",
    ),
    "job continue-on-error": (
        set_in(job_of, "continue-on-error", True),
        "forbidden key: continue-on-error",
    ),
    "step continue-on-error": (
        set_in(lambda d: step_of(d, 3), "continue-on-error", True),
        "forbidden key: continue-on-error",
    ),
    "environment": (
        set_in(job_of, "environment", "merge-guard"),
        "forbidden key: environment",
    ),
    "workflow concurrency": (
        set_in(
            lambda d: d,
            "concurrency",
            {"group": "guard", "cancel-in-progress": True},
        ),
        "forbidden key: concurrency",
    ),
    "job concurrency": (
        set_in(job_of, "concurrency", "guard"),
        "forbidden key: concurrency",
    ),
    "secret in env": (
        set_in(
            lambda d: step_of(d, 3),
            "env",
            {"MERGE_GUARD_TOKEN": "${{ secrets.GUARD_TOKEN }}"},
        ),
        "forbidden text: secrets.",
    ),
    "secrets inherit key": (
        set_in(job_of, "secrets", "inherit"),
        "forbidden key: secrets",
    ),
    "variable in env": (
        set_in(
            lambda d: step_of(d, 2),
            "env",
            {"BUNDLE_SHA256": "${{ vars.BUNDLE_SHA256 }}"},
        ),
        "forbidden text: vars.",
    ),
    "id-token permission": (
        set_in(lambda d: d, "permissions", PERMISSIONS | {"id-token": "write"}),
        "forbidden key: id-token",
    ),
    "job-level permissions": (
        set_in(job_of, "permissions", {"checks": "write"}),
        "job: keys are not",
    ),
    "write permission": (
        set_in(lambda d: d, "permissions", PERMISSIONS | {"checks": "write"}),
        "permissions: not exactly",
    ),
    "write-all permissions": (
        set_in(lambda d: d, "permissions", "write-all"),
        "permissions: not exactly",
    ),
    "a permission dropped": (
        set_in(lambda d: d, "permissions", {"contents": "read"}),
        "permissions: not exactly",
    ),
    "an extra read permission": (
        set_in(lambda d: d, "permissions", PERMISSIONS | {"issues": "read"}),
        "permissions: not exactly",
    ),
    "no permissions": (
        lambda d: d.pop("permissions"),
        "permissions: not exactly",
    ),
    "action from another owner": (
        set_in(lambda d: step_of(d, 1), "uses", "astral-sh/setup-uv@" + "a" * 40),
        "uses: a step is not an actions/ action",
    ),
    "action at a tag": (
        set_in(lambda d: step_of(d, 0), "uses", "actions/checkout@v6"),
        "uses: a step is not an actions/ action",
    ),
    "action at a short SHA": (
        set_in(lambda d: step_of(d, 0), "uses", "actions/checkout@de0fac2"),
        "uses: a step is not an actions/ action",
    ),
    "another pinned actions/ action": (
        set_in(lambda d: step_of(d, 0), "uses", "actions/cache@" + "a" * 40),
        "checkout: not the pinned checkout step",
    ),
    "checkout at another SHA": (
        set_in(lambda d: step_of(d, 0), "uses", "actions/checkout@" + "a" * 40),
        "checkout: not the pinned checkout step",
    ),
    "checkout of another repository": (
        set_in(
            lambda d: step_of(d, 0)["with"],
            "repository",
            "TheAxiomFoundation/rulespec-us",
        ),
        "checkout: repository is not axiom-encode",
    ),
    "checkout of the event repository": (
        lambda d: step_of(d, 0)["with"].pop("repository"),
        "checkout: inputs are not the reviewed set",
    ),
    "checkout of main": (
        set_in(lambda d: step_of(d, 0)["with"], "ref", "main"),
        "checkout: ref is not github.workflow_sha",
    ),
    "checkout of the event head": (
        set_in(
            lambda d: step_of(d, 0)["with"],
            "ref",
            "${{ github.event.pull_request.head.sha }}",
        ),
        "checkout: ref is not github.workflow_sha",
    ),
    "checkout keeps credentials": (
        set_in(lambda d: step_of(d, 0)["with"], "persist-credentials", True),
        "checkout: persist-credentials is not false",
    ),
    "checkout keeps credentials by default": (
        lambda d: step_of(d, 0)["with"].pop("persist-credentials"),
        "checkout: persist-credentials is not false",
    ),
    "checkout with a token input": (
        set_in(lambda d: step_of(d, 0)["with"], "token", "${{ github.token }}"),
        "forbidden key: token",
    ),
    "checkout with submodules": (
        set_in(lambda d: step_of(d, 0)["with"], "submodules", True),
        "checkout: inputs are not the reviewed set",
    ),
    "sparse checkout of one more file": (
        set_in(
            lambda d: step_of(d, 0)["with"],
            "sparse-checkout",
            "\n".join([*launch.SOURCE_FILES, "src/axiom_encode/cli.py"]) + "\n",
        ),
        "checkout: sparse-checkout is not the bundle list",
    ),
    "sparse checkout of one file less": (
        set_in(
            lambda d: step_of(d, 0)["with"],
            "sparse-checkout",
            "\n".join(launch.SOURCE_FILES[:-1]) + "\n",
        ),
        "checkout: sparse-checkout is not the bundle list",
    ),
    "sparse checkout of the package": (
        set_in(lambda d: step_of(d, 0)["with"], "sparse-checkout", "src\nscripts\n"),
        "checkout: sparse-checkout is not the bundle list",
    ),
    "full checkout": (
        lambda d: step_of(d, 0)["with"].pop("sparse-checkout"),
        "checkout: sparse-checkout is not the bundle list",
    ),
    "cone mode": (
        set_in(lambda d: step_of(d, 0)["with"], "sparse-checkout-cone-mode", True),
        "checkout: sparse-checkout-cone-mode is not false",
    ),
    "python cache": (
        set_in(lambda d: step_of(d, 1)["with"], "cache", "pip"),
        "forbidden key: cache",
    ),
    "another python": (
        set_in(lambda d: step_of(d, 1)["with"], "python-version", "3.x"),
        "python: not the pinned setup-python",
    ),
    "without -I": (
        set_in(
            lambda d: step_of(d, 3),
            "run",
            'set -euo pipefail\npython3 -S "$RUNNER_TEMP/guard/launch.py"\n',
        ),
        "evaluate: script is not exactly",
    ),
    "without -S": (
        set_in(
            lambda d: step_of(d, 3),
            "run",
            'set -euo pipefail\npython3 -I "$RUNNER_TEMP/guard/launch.py"\n',
        ),
        "evaluate: script is not exactly",
    ),
    "launcher result ignored": (
        set_in(
            lambda d: step_of(d, 3),
            "run",
            'set -euo pipefail\npython3 -I -S "$RUNNER_TEMP/guard/launch.py" || true\n',
        ),
        "evaluate: script is not exactly",
    ),
    "launcher piped": (
        set_in(
            lambda d: step_of(d, 3),
            "run",
            'set -euo pipefail\npython3 -I -S "$RUNNER_TEMP/guard/launch.py" | tee x\n',
        ),
        "forbidden text: | tee",
    ),
    "launcher with an argument": (
        set_in(
            lambda d: step_of(d, 3),
            "run",
            'set -euo pipefail\npython3 -I -S "$RUNNER_TEMP/guard/launch.py" --pr 1\n',
        ),
        "evaluate: script is not exactly",
    ),
    "console script instead of the launcher": (
        set_in(
            lambda d: step_of(d, 3),
            "run",
            "set -euo pipefail\nuv run axiom-encode merge-guard-enforce\n",
        ),
        "forbidden text: uv ",
    ),
    "evaluate without pipefail": (
        set_in(
            lambda d: step_of(d, 3),
            "run",
            'python3 -I -S "$RUNNER_TEMP/guard/launch.py"\n',
        ),
        "run: a script does not start with set -euo pipefail",
    ),
    "build without pipefail": (
        set_in(lambda d: step_of(d, 2), "run", "true\n"),
        "run: a script does not start with set -euo pipefail",
    ),
    "expression in a script": (
        set_in(
            lambda d: step_of(d, 2),
            "run",
            "set -euo pipefail\necho ${{ github.head_ref }}\n",
        ),
        "run: a script holds a workflow expression",
    ),
    "another expression": (
        set_in(
            lambda d: step_of(d, 3),
            "env",
            {"MERGE_GUARD_TOKEN": JOB_TOKEN, "X": "${{ github.head_ref }}"},
        ),
        "expressions: not exactly",
    ),
    "token from another place": (
        set_in(
            lambda d: step_of(d, 3),
            "env",
            {"MERGE_GUARD_TOKEN": "${{ github.event.inputs.token }}"},
        ),
        "evaluate: env is not exactly the job token",
    ),
    "repository override in env": (
        set_in(
            lambda d: step_of(d, 3),
            "env",
            {
                "MERGE_GUARD_TOKEN": JOB_TOKEN,
                "GITHUB_REPOSITORY": "TheAxiomFoundation/axiom-encode",
            },
        ),
        "evaluate: env is not exactly the job token",
    ),
    "job-level env": (
        set_in(job_of, "env", {"GITHUB_REPOSITORY": "TheAxiomFoundation/axiom-encode"}),
        "job: keys are not",
    ),
    "workflow-level env": (
        set_in(lambda d: d, "env", {"X": "1"}),
        "workflow: top-level keys",
    ),
    "digest literal of another bundle": (
        set_in(lambda d: step_of(d, 2), "env", {"BUNDLE_SHA256": "0" * 64}),
        "build: BUNDLE_SHA256 is",
    ),
    "digest not a literal digest": (
        set_in(lambda d: step_of(d, 2), "env", {"BUNDLE_SHA256": "main"}),
        "build: env is not exactly one BUNDLE_SHA256 literal",
    ),
    "digest step with another variable": (
        set_in(
            lambda d: step_of(d, 2),
            "env",
            {"BUNDLE_SHA256": "0" * 64, "SKIP_DIGEST": "1"},
        ),
        "build: env is not exactly one BUNDLE_SHA256 literal",
    ),
    "a fifth step": (
        lambda d: job_of(d)["steps"].append(
            {"name": "x", "run": "set -euo pipefail\n"}
        ),
        "steps: not exactly four steps",
    ),
    "the digest step removed": (
        lambda d: job_of(d)["steps"].pop(2),
        "steps: not exactly four steps",
    ),
    "a second job": (
        lambda d: d["jobs"].update(
            {"other": {"runs-on": "ubuntu-latest", "steps": [{"run": "true"}]}}
        ),
        "jobs: not exactly one job",
    ),
    "job renamed": (
        set_in(job_of, "name", "guard"),
        "job: name is not the guard check name",
    ),
    "workflow renamed": (
        set_in(lambda d: d, "name", "Merge guard"),
        "workflow: name is not the guard name",
    ),
    "self-hosted runner": (
        set_in(job_of, "runs-on", "self-hosted"),
        "job: runs-on is not ubuntu-latest",
    ),
    "timeout below the CI wait": (
        set_in(job_of, "timeout-minutes", 60),
        "job: timeout does not cover",
    ),
    "no timeout": (
        lambda d: job_of(d).pop("timeout-minutes"),
        "job: keys are not",
    ),
    "reusable workflow job": (
        set_in(job_of, "uses", "./.github/workflows/x.yml"),
        "job: keys are not",
    ),
    "matrix": (
        set_in(job_of, "strategy", {"matrix": {"x": [1, 2]}}),
        "forbidden key: strategy",
    ),
    "step shell": (
        set_in(lambda d: step_of(d, 3), "shell", "bash {0}"),
        "forbidden key: shell",
    ),
}


@pytest.mark.parametrize("case", sorted(NEGATIVE_CASES))
def test_the_tripwire_reports_each_change_to_the_workflow(case):
    change, expected = NEGATIVE_CASES[case]
    problems = problems_of(edited(change))
    assert any(problem.startswith(expected) for problem in problems), problems


def test_the_unchanged_workflow_survives_the_round_trip_the_cases_use():
    assert problems_of(edited(lambda document: None)) == []


@pytest.mark.parametrize("text", ["", "- a\n- b\n", "name: [unclosed\n", "x"])
def test_the_tripwire_reports_a_workflow_that_is_not_a_mapping(text):
    assert problems_of(text) != []


def test_the_tripwire_reports_a_bundle_change_without_a_new_literal():
    sources = bundle_sources()
    for path in launch.BUNDLE_FILES:
        changed = dict(sources)
        changed[path] = changed[path] + b"\n# one more line\n"
        assert bundle_digest(changed) != bundle_digest(sources), path
        problems = problems_of(workflow_text(), digest=bundle_digest(changed))
        assert any(p.startswith("build: BUNDLE_SHA256 is") for p in problems), path


def _job(**fields):
    return {"runs-on": "ubuntu-latest", "steps": [{"run": "true"}], **fields}


@pytest.mark.parametrize(
    ("others", "expected"),
    [
        (
            {"other.yml": {"name": "content-bound-merge-guard", "jobs": {"x": _job()}}},
            "workflow name can be the guard name",
        ),
        (
            {
                "other.yml": {
                    "name": " Content-Bound-Merge-Guard ",
                    "jobs": {"x": _job()},
                }
            },
            "workflow name can be the guard name",
        ),
        (
            {"other.yml": {"jobs": {"x": _job(name="content-bound-merge-guard")}}},
            "job name can publish the guard check name",
        ),
        (
            {"other.yml": {"jobs": {"x": _job(name="${{ inputs.name }}")}}},
            "job name can publish the guard check name",
        ),
        (
            {
                "other.yml": {
                    "jobs": {"x": _job(name="content-${{ matrix.kind }}-guard")}
                }
            },
            "job name can publish the guard check name",
        ),
        (
            {"other.yml": {"jobs": {"content-bound-merge-guard": _job()}}},
            "job id can publish the guard check name",
        ),
        (
            {
                "other.yml": {
                    "jobs": {
                        "call": {
                            "uses": "./.github/workflows/content-bound-merge-guard.yml"
                        }
                    }
                }
            },
            "calls the guard workflow",
        ),
        (
            {
                "other.yml": {
                    "jobs": {
                        "call": {
                            "uses": "TheAxiomFoundation/axiom-encode/.github/workflows/"
                            "content-bound-merge-guard.yml@main"
                        }
                    }
                }
            },
            "calls the guard workflow",
        ),
        (
            {"content-bound-merge-guard.yaml": {"name": "copy", "jobs": {"x": _job()}}},
            "a second file with the guard's name",
        ),
    ],
)
def test_the_tripwire_reports_another_workflow_that_takes_the_guards_place(
    others, expected
):
    problems = problems_of(workflow_text(), others=others)
    assert any(expected in problem for problem in problems), problems


@pytest.mark.parametrize(
    "others",
    [
        {"other.yml": {"name": "CI", "jobs": {"test": _job(name="test (3.13)")}}},
        {
            "other.yml": {
                "name": "Merge guard runner (shadow)",
                "jobs": {"run": _job(name="test (${{ matrix.python-version }})")},
            }
        },
        {"other.yml": {"jobs": {"call": {"uses": "./.github/workflows/helper.yml"}}}},
    ],
)
def test_the_tripwire_accepts_unrelated_workflows(others):
    assert problems_of(workflow_text(), others=others) == []


# --- the bundle: standard library and listed modules only -----------------------

ALLOWED_PACKAGE_NAMES = frozenset(launch.BUNDLE_MODULES)


def import_problems(source: str, name: str) -> list[str]:
    """Imports of one bundle file that leave the stdlib and the bundle list.

    Walks the whole syntax tree, so an import inside a function, a class, a
    ``try`` or an ``if`` counts like one at module level.
    """
    problems = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            for alias in node.names:
                parts = alias.name.split(".")
                if parts[0] == launch.PACKAGE:
                    if len(parts) != 2 or parts[1] not in ALLOWED_PACKAGE_NAMES:
                        problems.append(f"{name}: imports {alias.name}")
                elif parts[0] not in sys.stdlib_module_names:
                    problems.append(f"{name}: imports {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            parts = module.split(".")
            if node.level:
                problems.append(f"{name}: relative import of {module or '.'}")
            elif module == launch.PACKAGE:
                for alias in node.names:
                    if alias.name not in ALLOWED_PACKAGE_NAMES:
                        problems.append(f"{name}: imports {module}.{alias.name}")
            elif parts[0] == launch.PACKAGE:
                if len(parts) != 2 or parts[1] not in ALLOWED_PACKAGE_NAMES:
                    problems.append(f"{name}: imports {module}")
            elif parts[0] not in sys.stdlib_module_names:
                problems.append(f"{name}: imports {module}")
        elif isinstance(node, ast.Name) and node.id == "__import__":
            problems.append(f"{name}: calls __import__")
        elif isinstance(node, ast.Attribute) and node.attr in {
            "import_module",
            "__import__",
            "exec_module",
            "load_module",
        }:
            problems.append(f"{name}: imports dynamically ({node.attr})")
        elif isinstance(node, ast.Name) and node.id in {"exec", "eval"}:
            problems.append(f"{name}: calls {node.id}")
    return problems


def test_the_bundle_list_is_the_reviewed_list():
    assert launch.BUNDLE_MODULES == (
        "applied_manifest_v6",
        "content_merge_guard",
        "merge_guard_runner",
        "reviewed_pr_admission",
        "rulespec_citation_paths",
        "statute",
    )
    assert launch.SOURCE_FILES == (
        "scripts/merge_guard_launch.py",
        "src/axiom_encode/applied_manifest_v6.py",
        "src/axiom_encode/content_merge_guard.py",
        "src/axiom_encode/merge_guard_runner.py",
        "src/axiom_encode/reviewed_pr_admission.py",
        "src/axiom_encode/rulespec_citation_paths.py",
        "src/axiom_encode/statute.py",
    )
    assert launch.BUNDLE_FILES == (
        "launch.py",
        "axiom_encode/__init__.py",
        "axiom_encode/applied_manifest_v6.py",
        "axiom_encode/content_merge_guard.py",
        "axiom_encode/merge_guard_runner.py",
        "axiom_encode/reviewed_pr_admission.py",
        "axiom_encode/rulespec_citation_paths.py",
        "axiom_encode/statute.py",
    )
    assert launch.ALLOWED_MODULES == {"axiom_encode"} | {
        "axiom_encode." + module for module in launch.BUNDLE_MODULES
    }


@pytest.mark.parametrize("path", launch.SOURCE_FILES)
def test_each_bundle_file_imports_only_the_stdlib_and_the_bundle(path):
    source = (REPOSITORY_ROOT / path).read_text(encoding="utf-8")
    assert import_problems(source, path) == []


def test_the_bundle_is_closed_under_its_own_imports():
    # Every bundle module that a bundle file imports is itself on the list,
    # and every listed module is reached from the runner.
    reached, pending = set(), ["merge_guard_runner"]
    while pending:
        module = pending.pop()
        if module in reached:
            continue
        reached.add(module)
        tree = ast.parse(
            (REPOSITORY_ROOT / "src" / launch.PACKAGE / f"{module}.py").read_text()
        )
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                parts = node.module.split(".")
                if node.module == launch.PACKAGE:
                    pending.extend(alias.name for alias in node.names)
                elif parts[0] == launch.PACKAGE:
                    pending.append(parts[1])
    assert reached == set(launch.BUNDLE_MODULES)


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("import yaml\n", "imports yaml"),
        ("import os, requests\n", "imports requests"),
        ("from pydantic import BaseModel\n", "imports pydantic"),
        ("def f():\n    import yaml\n", "imports yaml"),
        ("def f():\n    from axiom_encode import cli\n", "imports axiom_encode.cli"),
        (
            "try:\n    import tomli\nexcept ImportError:\n    tomli = None\n",
            "imports tomli",
        ),
        ("if True:\n    from yaml import safe_load\n", "imports yaml"),
        ("class A:\n    def f(self):\n        import httpx.client\n", "imports httpx"),
        ("from axiom_encode import cli\n", "imports axiom_encode.cli"),
        ("from axiom_encode.cli import main\n", "imports axiom_encode.cli"),
        ("import axiom_encode.cli\n", "imports axiom_encode.cli"),
        ("import axiom_encode\n", "imports axiom_encode"),
        ("from axiom_encode.harness import evals\n", "imports axiom_encode.harness"),
        ("from axiom_encode.statute.x import y\n", "imports axiom_encode.statute.x"),
        ("from . import cli\n", "relative import"),
        ("from .cli import main\n", "relative import"),
        ("import importlib\nimportlib.import_module('yaml')\n", "imports dynamically"),
        ("__import__('yaml')\n", "calls __import__"),
        ("exec('import yaml')\n", "calls exec"),
    ],
)
def test_the_import_check_reports_each_way_out_of_the_bundle(source, expected):
    problems = import_problems(source, "x.py")
    assert any(expected in problem for problem in problems), problems


@pytest.mark.parametrize(
    "source",
    [
        "import json\nimport urllib.request\n",
        "from axiom_encode import content_merge_guard as guard\n",
        "from axiom_encode.statute import citation_to_citation_path\n",
        "def f():\n    import hashlib\n    return hashlib\n",
    ],
)
def test_the_import_check_accepts_stdlib_and_listed_modules(source):
    assert import_problems(source, "x.py") == []


# --- the launcher's finder, in this process -------------------------------------


@pytest.mark.parametrize(
    "name",
    [
        "axiom_encode.cli",
        "axiom_encode.entrypoint",
        "axiom_encode.signing_broker",
        "axiom_encode.harness",
        "axiom_encode.harness.evals",
        "axiom_encode.statute.inner",
        "axiom_encode.merge_guard_runner_extra",
    ],
)
def test_the_finder_refuses_every_unlisted_encoder_module(name):
    with pytest.raises(ImportError, match="not a module of the merge guard bundle"):
        launch._RefuseUnlistedModules.find_spec(name)


@pytest.mark.parametrize(
    "name",
    [
        *sorted(launch.ALLOWED_MODULES),
        "json",
        "urllib.request",
        "axiom_encode_other",
        "axiom_encodex.cli",
    ],
)
def test_the_finder_leaves_listed_and_other_modules_to_the_standard_finders(name):
    assert launch._RefuseUnlistedModules.find_spec(name) is None


def test_the_launcher_refuses_this_process_which_is_not_isolated():
    assert not (sys.flags.isolated and sys.flags.no_site)
    with pytest.raises(launch.LaunchError, match="-I -S"):
        launch.prepare()
    # Nothing was installed by the refused call.
    assert not any(
        isinstance(finder, launch._RefuseUnlistedModules) for finder in sys.meta_path
    )


def test_loaded_outside_names_encoder_modules_of_this_process(tmp_path):
    # This process loaded the encoder package from ``src``: none of it is in
    # ``tmp_path``.
    problems = launch.loaded_outside(str(tmp_path))
    assert "axiom_encode" in problems
    assert "axiom_encode.merge_guard_runner" in problems
    assert "axiom_encode.cli" in problems
    assert all(name.split(".")[0] == "axiom_encode" for name in problems)


# --- the build step and the launcher, as the job runs them -----------------------


def build_script() -> tuple[str, dict]:
    step = workflow()["jobs"]["guard"]["steps"][2]
    return step["run"], step["env"]


def run_build_step(tmp_path: Path, source_root: Path, **env):
    """Run the workflow's build step with ``bash``; returns the process."""
    workspace = tmp_path / "workspace"
    workspace.mkdir(exist_ok=True)
    link = workspace / "guard-source"
    if not link.exists():
        link.symlink_to(source_root, target_is_directory=True)
    script, step_env = build_script()
    return subprocess.run(
        ["bash", "-c", script],
        env={
            "PATH": os.environ["PATH"],
            "GITHUB_WORKSPACE": str(workspace),
            "RUNNER_TEMP": str(tmp_path / "runner-temp"),
        }
        | step_env
        | env,
        capture_output=True,
        text=True,
        timeout=60,
    )


needs_sha256sum = pytest.mark.skipif(
    shutil.which("sha256sum") is None or shutil.which("bash") is None,
    reason="the build step needs bash and sha256sum, as on the hosted runner",
)


@pytest.fixture
def bundle(tmp_path):
    """The bundle directory as the workflow's own build step makes it."""
    if shutil.which("sha256sum") is None or shutil.which("bash") is None:
        pytest.skip("the build step needs bash and sha256sum, as on the hosted runner")
    (tmp_path / "runner-temp").mkdir()
    result = run_build_step(tmp_path, REPOSITORY_ROOT)
    assert result.returncode == 0, result.stdout + result.stderr
    return tmp_path / "runner-temp" / "guard"


def listing(directory: Path) -> list[str]:
    return sorted(
        path.relative_to(directory).as_posix()
        for path in directory.rglob("*")
        if path.is_file()
    )


def test_the_build_step_makes_exactly_the_bundle(bundle):
    assert listing(bundle) == sorted(launch.BUNDLE_FILES)
    assert {path: (bundle / path).read_bytes() for path in launch.BUNDLE_FILES} == (
        bundle_sources()
    )


@needs_sha256sum
def test_the_build_step_computes_the_digest_this_file_defines(tmp_path):
    (tmp_path / "runner-temp").mkdir()
    wrong = run_build_step(tmp_path, REPOSITORY_ROOT, BUNDLE_SHA256="0" * 64)
    assert wrong.returncode == 1
    assert bundle_digest(bundle_sources()) in wrong.stdout
    assert "::error::" in wrong.stdout


@needs_sha256sum
@pytest.mark.parametrize("path", launch.SOURCE_FILES)
def test_the_build_step_refuses_a_bundle_with_one_changed_file(tmp_path, path):
    source = tmp_path / "changed-source"
    for relative in launch.SOURCE_FILES:
        (source / relative).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPOSITORY_ROOT / relative, source / relative)
    with (source / path).open("ab") as stream:
        stream.write(b"\n")
    (tmp_path / "runner-temp").mkdir()
    result = run_build_step(tmp_path, source)
    assert result.returncode == 1, result.stdout + result.stderr
    assert "::error::The guard bundle digest is" in result.stdout


@needs_sha256sum
def test_the_build_step_refuses_a_missing_file_and_an_existing_directory(tmp_path):
    source = tmp_path / "partial-source"
    for relative in launch.SOURCE_FILES[:-1]:
        (source / relative).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPOSITORY_ROOT / relative, source / relative)
    (tmp_path / "runner-temp").mkdir()
    assert run_build_step(tmp_path, source).returncode != 0
    # The directory of the failed build is still there: a second build refuses.
    again = run_build_step(tmp_path, REPOSITORY_ROOT)
    assert again.returncode == 1
    assert "already exists" in again.stdout


def launcher(bundle: Path, *flags: str, **env):
    return subprocess.run(
        [sys.executable, *flags, str(bundle / "launch.py")],
        env={"PATH": os.environ["PATH"]} | env,
        capture_output=True,
        text=True,
        timeout=120,
        cwd=bundle.parent,
    )


SOURCE_ENV = {
    "GITHUB_REPOSITORY": "TheAxiomFoundation/axiom-encode",
    "GITHUB_EVENT_NAME": "pull_request",
}


def test_the_launcher_is_inert_in_the_source_repository(bundle):
    result = launcher(bundle, "-I", "-S", **SOURCE_ENV)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {
        "schema": runner.ENFORCE_REPORT_SCHEMA,
        "event": "pull_request",
        "repository": "TheAxiomFoundation/axiom-encode",
        "verdict": None,
        "requests": 0,
        "inert": "source_repository",
    }
    assert listing(bundle) == sorted(launch.BUNDLE_FILES)  # no __pycache__


@pytest.mark.parametrize(
    "env",
    [
        {"GITHUB_REPOSITORY": REPO, "GITHUB_EVENT_NAME": "pull_request"},
        {
            "GITHUB_REPOSITORY": REPO,
            "GITHUB_EVENT_NAME": "pull_request",
            "MERGE_GUARD_TOKEN": TOKEN,
        },
        {
            "GITHUB_REPOSITORY": REPO,
            "GITHUB_EVENT_NAME": "pull_request",
            "MERGE_GUARD_TOKEN": TOKEN,
            "GITHUB_EVENT_PATH": "missing.json",
        },
        {"GITHUB_REPOSITORY": "TheAxiomFoundation/axiom-encode-fork"},
        {},
    ],
)
def test_the_launcher_fails_outside_the_source_repository_without_inputs(bundle, env):
    # No case here reaches the network: each stops at the token or the event.
    result = launcher(bundle, "-I", "-S", **env)
    assert result.returncode == 1
    report = json.loads(result.stdout)
    assert report["verdict"] is None and "inert" not in report
    assert TOKEN not in result.stdout + result.stderr


def test_the_launcher_fails_on_a_malformed_event(bundle, tmp_path):
    event = tmp_path / "event.json"
    event.write_text('{"number": 42, "number": 42}')
    result = launcher(
        bundle,
        "-I",
        "-S",
        GITHUB_REPOSITORY=REPO,
        GITHUB_EVENT_NAME="pull_request",
        MERGE_GUARD_TOKEN=TOKEN,
        GITHUB_EVENT_PATH=str(event),
    )
    assert result.returncode == 1
    assert "not strict JSON" in json.loads(result.stdout)["error"]


@pytest.mark.parametrize("flags", [(), ("-I",), ("-S",), ("-E", "-s", "-S")])
def test_the_launcher_refuses_an_interpreter_that_is_not_isolated(bundle, flags):
    result = launcher(bundle, *flags, **SOURCE_ENV)
    assert result.returncode == 1
    assert result.stdout == ""
    assert "python -I -S only" in result.stderr


@pytest.mark.parametrize(
    ("path", "content"),
    [
        ("axiom_encode/cli.py", b"VALUE = 1\n"),
        ("axiom_encode/harness/__init__.py", b""),
        ("json.py", b"raise SystemExit(0)\n"),
        ("sitecustomize.py", b""),
        ("axiom_encode/statute.pyc", b""),
        ("notes.txt", b""),
    ],
)
def test_the_launcher_refuses_a_bundle_directory_with_one_more_file(
    bundle, path, content
):
    (bundle / path).parent.mkdir(parents=True, exist_ok=True)
    (bundle / path).write_bytes(content)
    result = launcher(bundle, "-I", "-S", **SOURCE_ENV)
    assert result.returncode == 1 and result.stdout == ""
    assert "does not hold exactly the bundle" in result.stderr


def test_the_launcher_refuses_a_bundle_directory_with_one_file_less(bundle):
    (bundle / "axiom_encode" / "statute.py").unlink()
    result = launcher(bundle, "-I", "-S", **SOURCE_ENV)
    assert result.returncode == 1 and result.stdout == ""
    assert "does not hold exactly the bundle" in result.stderr


def test_the_launcher_refuses_a_package_init_that_is_not_empty(bundle):
    (bundle / "axiom_encode" / "__init__.py").write_text("import os\n")
    result = launcher(bundle, "-I", "-S", **SOURCE_ENV)
    assert result.returncode == 1 and result.stdout == ""
    assert "__init__.py is not empty" in result.stderr


def test_the_launcher_refuses_a_link_in_the_bundle_directory(bundle, tmp_path):
    target = bundle / "axiom_encode" / "statute.py"
    moved = tmp_path / "statute.py"
    shutil.move(target, moved)
    target.symlink_to(moved)
    result = launcher(bundle, "-I", "-S", **SOURCE_ENV)
    assert result.returncode == 1 and result.stdout == ""
    assert "holds a link" in result.stderr


def test_the_launcher_fails_closed_when_the_runner_raises(bundle):
    # A bundle whose runner cannot be imported: the job fails, with no report.
    (bundle / "axiom_encode" / "merge_guard_runner.py").write_text(
        "raise RuntimeError('broken')\n"
    )
    result = launcher(bundle, "-I", "-S", **SOURCE_ENV)
    assert result.returncode == 1 and result.stdout == ""
    assert "merge guard launcher failed: RuntimeError" in result.stderr


def driver(*arguments: str):
    result = subprocess.run(
        [sys.executable, "-I", "-S", str(DRIVER_PATH), *arguments],
        env={"PATH": os.environ["PATH"]},
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def test_the_launcher_refuses_an_encoder_package_from_outside_the_bundle(
    bundle, tmp_path
):
    # Another ``axiom_encode`` package comes first on sys.path and its runner
    # would end the job with success. Its modules are on the list by name, so
    # the finder admits them; the check of where they were loaded from must
    # refuse.
    foreign = tmp_path / "foreign"
    (foreign / "axiom_encode").mkdir(parents=True)
    (foreign / "axiom_encode" / "__init__.py").write_text("")
    (foreign / "axiom_encode" / "merge_guard_runner.py").write_text(
        "def enforce_command():\n    print('forged')\n    return 0\n"
    )
    result = driver("foreign", str(bundle), str(foreign))
    assert result["code"] == 1
    assert "forged" not in result["output"]
    assert "loaded from outside the bundle" in result["output"]


def sweep_cases(tmp_path: Path) -> list[dict]:
    """One enforce run per lane, as data the driver can replay."""

    def case(name, world, event, event_name, sequences=None):
        event_path = tmp_path / f"{name}.event.json"
        event_path.write_text(json.dumps(event))
        api = {}
        for path, value in world.api.items():
            assert not callable(value), path
            api[path] = [copy.deepcopy(value)]
        api |= sequences or {}
        return {
            "name": name,
            "environ": {
                "GITHUB_REPOSITORY": REPO,
                "GITHUB_EVENT_NAME": event_name,
                "GITHUB_EVENT_PATH": str(event_path),
                runner.TOKEN_ENV: TOKEN,
            },
            "api": api,
        }

    unapproved = build()
    unapproved.reviews([])
    waited = build()
    listing_path = f"{ROOT}/commits/{H}{CHECKS}"
    green = copy.deepcopy(waited.api[listing_path])
    waited.ci(H, status="in_progress", conclusion=None)
    pending = copy.deepcopy(waited.api[listing_path])
    return [
        case("v6_pass", build(), pr_event(), "pull_request"),
        case("v5_lane", plain(V5_LANE), pr_event(), "pull_request"),
        case("allowlist", plain({"README.md": b"x\n"}), pr_event(), "pull_request"),
        case(
            "no_lane",
            plain({".github/workflows/x.yml": b"x\n"}),
            pr_event(),
            "pull_request",
        ),
        case("unapproved", unapproved, pr_event(), "pull_request"),
        case(
            "waited",
            waited,
            pr_event(),
            "pull_request",
            {listing_path: [pending, pending, green]},
        ),
        case("merge_group", queue(build()), group_event(), "merge_group"),
        case(
            "merge_group_two_units",
            queue(build(), group_tree="f" * 40),
            group_event(),
            "merge_group",
        ),
        case("stale_head", build(), pr_event(head="a" * 40), "pull_request"),
        case("unreadable", plain({}), pr_event(number=43), "pull_request"),
    ]


EXPECTED_SWEEP = {
    "v6_pass": (0, "pass", guard.R_PASS),
    "v5_lane": (0, "not_applicable", guard.R_EARLIER_LANE),
    "allowlist": (0, "not_applicable", guard.R_NO_GUARDED_CONTENT),
    "no_lane": (1, "fail", guard.R_NO_LANE),
    "unapproved": (1, "fail", guard.R_NO_APPROVAL),
    "waited": (0, "pass", guard.R_PASS),
    "merge_group": (0, "pass", guard.R_PASS),
    "merge_group_two_units": (1, "fail", guard.R_MERGE_GROUP_TREE),
    "stale_head": (1, "fail", runner.R_EVENT_HEAD_STALE),
    "unreadable": (1, "fail", guard.R_READ_FAILED),
}


def test_a_sweep_of_every_lane_under_the_launcher_loads_only_the_bundle(
    bundle, tmp_path
):
    scenario = tmp_path / "scenario.json"
    scenario.write_text(json.dumps({"cases": sweep_cases(tmp_path)}))
    result = driver("sweep", str(bundle), str(scenario))

    assert result["flags"] == [1, 1]
    assert result["prepared"] == str(bundle)
    outcomes = {
        item["name"]: (
            item["code"],
            item["report"]["verdict"]["verdict"],
            item["report"]["verdict"]["reason"],
        )
        for item in result["results"]
    }
    assert outcomes == EXPECTED_SWEEP
    waited = next(item for item in result["results"] if item["name"] == "waited")
    assert waited["report"]["evaluations"] == 2 and waited["report"]["ci_polls"] == 2
    # Reads only, through the real HTTP layer of the runner.
    for item in result["results"]:
        assert item["methods"] == ["GET"], item["name"]
        assert item["bodies"] == 0, item["name"]

    loaded = result["modules"]
    encoder = {name for name in loaded if name.split(".")[0] == "axiom_encode"}
    assert encoder == launch.ALLOWED_MODULES
    foreign = sorted(
        name
        for name in loaded
        if name.split(".")[0] not in sys.stdlib_module_names
        and name.split(".")[0] != "axiom_encode"
        and name not in {"__main__", "merge_guard_launch_under_test"}
    )
    assert foreign == []
    assert result["loaded_outside"] == []
    assert result["bundle_is_last_on_path"] is True
    assert result["pycache"] is False
    # The finder refuses an unlisted module that IS in the bundle directory,
    # and modules of the encoder that are not.
    for name, message in result["refusals"].items():
        assert message == f"{name} is not a module of the merge guard bundle"
    assert set(result["refusals"]) == {
        "axiom_encode.unlisted",
        "axiom_encode.cli",
        "axiom_encode.harness",
    }


def test_the_driver_is_a_stdlib_only_helper_outside_the_bundle():
    assert sorted(set(import_problems(DRIVER_PATH.read_text(), "driver"))) == [
        "driver: imports dynamically (exec_module)",
        "driver: imports dynamically (import_module)",
    ]
    assert "tests/" + DRIVER_PATH.name not in launch.SOURCE_FILES


def test_the_guard_files_have_both_code_owners():
    # The workflow carries the digest of the bundle, so a bundle change needs
    # a change to a file under ``.github/``; these entries give the launcher,
    # the workflow and their tripwires an owner by name as well.
    lines = (REPOSITORY_ROOT / ".github" / "CODEOWNERS").read_text().splitlines()
    for path in (
        "/.github/workflows/content-bound-merge-guard.yml",
        "/scripts/merge_guard_launch.py",
        "/tests/test_merge_guard_enforce_workflow.py",
        "/tests/merge_guard_bundle_driver.py",
        *("/" + source for source in launch.SOURCE_FILES),
    ):
        assert f"{path} @PavelMakarchuk @MaxGhenis" in lines, path
