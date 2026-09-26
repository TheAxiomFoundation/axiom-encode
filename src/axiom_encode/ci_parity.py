"""Local, verification-only parity entrypoint for validate-rulespec CI.

This module deliberately owns no signing capability and never invokes an applying
command. Apart from caching a fetched immutable public release object at the
workflow-defined corpus path, it writes only temporary report inputs. Protected-supervisor calls
in CI are reproduced as direct subcommands under an explicit, library-level
corpus release verification keyring.

Workflow pins from 0effa6a5 on are executed from the pinned workflow itself: the
byte-exact reusable workflow ships in ``ci_parity_workflows/`` (verified against
its Git blob id), and every inline Python script a gate runs in CI is extracted
from it and executed unchanged. Only the bash glue around those scripts, and the
supervised ``axiom-encode`` subcommands, are reproduced here.
"""

from __future__ import annotations

import argparse
import contextlib
import fnmatch
import hashlib
import importlib.metadata
import io
import json
import os
import re
import shlex
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import tomllib
import unicodedata
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

from axiom_encode import __version__
from axiom_encode.corpus_resolver import MAX_RELEASE_OBJECT_BYTES
from axiom_encode.toolchain import (
    RuleSpecToolchain,
    load_rulespec_local_corpus_release,
    load_rulespec_toolchain,
    local_corpus_release_verification,
    verify_rulespec_validation_waiver_set,
)

SHA_RE = re.compile(r"^[0-9a-f]{40}$")
WORKFLOW_RE = re.compile(
    r"TheAxiomFoundation/\.github/\.github/workflows/"
    r"validate-rulespec\.yml@(?P<sha>[0-9a-f]{40})$"
)
DEPENDENCY_INPUTS = {
    "encode": "axiom-encode-ref",
    "engine": "axiom-rules-engine-ref",
    "corpus": "axiom-corpus-ref",
    "rulespec_us": "rulespec-us-ref",
}
DEFAULT_RELEASE_BASE_URL = "https://pub-a8952f8657fc49fda358146ac001366c.r2.dev"
# The organization's public corpus.release_objects registry. Callers pass it
# from the NEXT_PUBLIC_SUPABASE_URL repository variable, which a local run
# cannot read; --corpus-release-registry-url supplies another value.
DEFAULT_RELEASE_REGISTRY_URL = "https://swocpijqqahhuwtuahwc.supabase.co"
# The anon key is a public read credential, not a trust root: every fetched
# object is pinned by content hash and signature-verified before use, so an
# environment default is safe for it (unlike --corpus-release-public-key).
RELEASE_REGISTRY_ANON_KEY_ENV = "NEXT_PUBLIC_SUPABASE_ANON_KEY"
# The repository variable callers pass as corpus-release-registry-url. Only a
# caller reading this variable gets DEFAULT_RELEASE_REGISTRY_URL by default.
RELEASE_REGISTRY_URL_VARIABLE = "NEXT_PUBLIC_SUPABASE_URL"
# A bound on a single release object download. The resolver enforces its own,
# smaller cap on the rewritten object; this only keeps a hostile or broken
# endpoint from exhausting local memory.
MAX_RELEASE_DOWNLOAD_BYTES = 4 * MAX_RELEASE_OBJECT_BYTES
WORKFLOW_DIRECTORY = Path(__file__).with_name("ci_parity_workflows")
PULL_REQUEST_EVENT = "pull_request"
EXPRESSION_RE = re.compile(r"\$\{\{\s*(?P<expression>[^{}]*?)\s*\}\}")
NEEDS_OUTPUT_RE = re.compile(
    r"needs\.(?P<job>[A-Za-z_][A-Za-z0-9_-]*)\.outputs\.(?P<output>[A-Za-z_][A-Za-z0-9_-]*)"
)
STEP_OUTPUT_RE = re.compile(
    r"steps\.(?P<step>[A-Za-z_][A-Za-z0-9_-]*)\.outputs\.(?P<output>[A-Za-z_][A-Za-z0-9_-]*)"
)
VARS_RE = re.compile(r"vars\.(?P<name>[A-Za-z_][A-Za-z0-9_]*)")
# Reviewed caller-side resolver steps that turn `.axiom/workflow-toolchain.toml`
# into `needs.<job>.outputs.*` dependency refs, keyed by the sha256 of the
# step's `run` script. ci executes a recognized script unchanged; any other
# resolver fails closed until it is reviewed and added here.
RECOGNIZED_WORKFLOW_TOOLCHAIN_RESOLVERS: dict[str, str] = {
    "677a17ea1e348bd0d564095ae29c3ea9f4cd00f0ba9ee37302a5407bcc315da3": (
        "TheAxiomFoundation/rulespec-us repository-checks.yml@cf03a179"
    ),
}
# Environment names the protected verification supervisor forwards into its
# otherwise empty child environment (cmd/axiom-encode-signing-supervisor
# parentOnlyEnvironmentNames). Other AXIOM_* values never reach a supervised
# CI gate, so ci withholds them from the in-process equivalent.
SUPERVISOR_FORWARDED_ENVIRONMENT = frozenset(
    {
        "AXIOM_ENCODE_APPLY_CHECKOUT",
        "AXIOM_ENCODE_SUPABASE_URL",
        "AXIOM_ENCODE_SUPABASE_SECRET_KEY",
        "AXIOM_ENCODE_SUPABASE_ANON_KEY",
    }
)
SUPERVISOR_GIT_ENVIRONMENT = {
    "GIT_CONFIG_GLOBAL": os.devnull,
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_TERMINAL_PROMPT": "0",
}
WAIVER_AUDIT_WORKERS_ENV = "AXIOM_ENCODE_WAIVER_AUDIT_WORKERS"


@dataclass(frozen=True, slots=True)
class WorkflowInput:
    type: str
    required: bool = False
    default: str | bool | int | None = None


# workflow_call inputs, exactly as each pinned workflow declares them; the
# parity tests compare these tables with the packaged workflows.
LEGACY_WORKFLOW_INPUTS: dict[str, WorkflowInput] = {
    "python-version": WorkflowInput("string", default="3.14"),
    "axiom-encode-ref": WorkflowInput("string", required=True),
    "axiom-rules-engine-ref": WorkflowInput("string", required=True),
    "axiom-corpus-ref": WorkflowInput("string", required=True),
    "rulespec-us-ref": WorkflowInput("string", required=True),
    "corpus-release-base-url": WorkflowInput(
        "string", default=DEFAULT_RELEASE_BASE_URL
    ),
    "validate-roots": WorkflowInput("string", default="statutes regulations policies"),
    "run-pytest": WorkflowInput("boolean", default=True),
    "run-generated-guard": WorkflowInput("boolean", default=True),
    "guard-programs-root": WorkflowInput("boolean", default=False),
    "run-money-atom-check": WorkflowInput("boolean", default=True),
}
WORKFLOW_INPUTS_0EFFA6A5: dict[str, WorkflowInput] = {
    "python-version": WorkflowInput("string", default="3.14"),
    "axiom-encode-ref": WorkflowInput("string", required=True),
    "axiom-rules-engine-ref": WorkflowInput("string", required=True),
    "axiom-corpus-ref": WorkflowInput("string", required=True),
    "rulespec-us-ref": WorkflowInput("string", required=True),
    "corpus-release-base-url": WorkflowInput(
        "string", default=DEFAULT_RELEASE_BASE_URL
    ),
    "corpus-release-registry-url": WorkflowInput("string", default=""),
    "corpus-release-registry-anon-key": WorkflowInput("string", default=""),
    "validate-roots": WorkflowInput("string", default="statutes regulations policies"),
    "validation-workers": WorkflowInput("number", default=1),
    "run-pytest": WorkflowInput("boolean", default=True),
    "run-generated-guard": WorkflowInput("boolean", default=True),
    "migration-authorization-path": WorkflowInput("string", default=""),
    "retired-schema-bootstrap-sha256": WorkflowInput("string", default=""),
    "allow-retired-schema-prefreeze": WorkflowInput("boolean", default=False),
    "validation-waiver-bootstrap-sha256": WorkflowInput("string", default=""),
    "guard-programs-root": WorkflowInput("boolean", default=False),
    "run-money-atom-check": WorkflowInput("boolean", default=True),
}
_INPUT_TYPE_DEFAULTS: dict[str, str | bool | int] = {
    "string": "",
    "boolean": False,
    "number": 0,
}


@dataclass(frozen=True, slots=True)
class WorkflowGateParameters:
    exclude_programs_from_money_atom_check: bool
    # Pins from 0effa6a5 on run execute_workflow_gates, which executes the
    # pinned workflow's own inline scripts; older pins keep execute_gates.
    embedded_scripts: bool = False
    # The shards-job "Reject unmanifested RuleSpec content" pre-check (6f11be26).
    unmanifested_precheck: bool = False
    # The generated guard refuses a base ref that is not an exact 40-hex SHA.
    exact_guard_base_ref: bool = False
    # The supervisor is provisioned with the retired corpus release root too.
    retired_corpus_release_key: bool = False


@dataclass(frozen=True, slots=True)
class SupportedWorkflowPin:
    fixture: str
    gate_parameters: WorkflowGateParameters
    # Git blob id of the packaged workflow file, i.e.
    # `git rev-parse <pin>:.github/workflows/validate-rulespec.yml`.
    workflow_blob: str
    inputs: Mapping[str, WorkflowInput]
    gates: tuple[str, ...]


LEGACY_GATE_ORDER: tuple[str, ...] = (
    "repository_tests",
    "obsolete_files",
    "repository_layout",
    "validation_waivers",
    "guard_generated",
    "select_targets",
    "validate",
    "companion_tests",
    "proof_validate",
    "money_atoms",
    "oracle_coverage",
    "changed_oracle_coverage",
)
WORKFLOW_GATE_ORDER_0EFFA6A5: tuple[str, ...] = (
    "unsupported_paths",
    "migration_authorization",
    "retired_schema_freeze",
    "obsolete_files",
    "repository_layout",
    "validation_waivers",
    "guard_generated",
    "select_targets",
    "validate",
    "companion_tests",
    "proof_validate",
    "money_atoms",
    "oracle_coverage",
    "changed_oracle_coverage",
    "repository_tests",
)
WORKFLOW_GATE_ORDER_6F11BE26: tuple[str, ...] = (
    "unsupported_paths",
    "unmanifested_rulespec",
    *WORKFLOW_GATE_ORDER_0EFFA6A5[1:],
)

SUPPORTED_WORKFLOW_PINS: dict[str, SupportedWorkflowPin] = {
    "615c1df9b9ace7deea84da65efd137f46f8bad2b": SupportedWorkflowPin(
        fixture="validate-rulespec-615c1df9.yml",
        gate_parameters=WorkflowGateParameters(
            exclude_programs_from_money_atom_check=True
        ),
        workflow_blob="a4df9e7948a438a91112621ddaa46f5e4ad7d76c",
        inputs=LEGACY_WORKFLOW_INPUTS,
        gates=LEGACY_GATE_ORDER,
    ),
    "34bcfab235c585c47292c95f51be1a4f4f91d29e": SupportedWorkflowPin(
        fixture="validate-rulespec-34bcfab2.yml",
        gate_parameters=WorkflowGateParameters(
            exclude_programs_from_money_atom_check=False
        ),
        workflow_blob="28bbe173a9f289b567cd75afa95aaea4c15fb836",
        inputs=LEGACY_WORKFLOW_INPUTS,
        gates=LEGACY_GATE_ORDER,
    ),
    "0effa6a5b05e7fac53902df7d523e909bd7fc48a": SupportedWorkflowPin(
        fixture="validate-rulespec-0effa6a5.yml",
        gate_parameters=WorkflowGateParameters(
            exclude_programs_from_money_atom_check=True,
            embedded_scripts=True,
            retired_corpus_release_key=True,
        ),
        workflow_blob="0224ea4f9cdb944cb4c389e21d6a2f99c17dc820",
        inputs=WORKFLOW_INPUTS_0EFFA6A5,
        gates=WORKFLOW_GATE_ORDER_0EFFA6A5,
    ),
    "6f11be2655f79dd0a3b582db46525f58332ca120": SupportedWorkflowPin(
        fixture="validate-rulespec-6f11be26.yml",
        gate_parameters=WorkflowGateParameters(
            exclude_programs_from_money_atom_check=True,
            embedded_scripts=True,
            unmanifested_precheck=True,
            exact_guard_base_ref=True,
            retired_corpus_release_key=True,
        ),
        workflow_blob="cd5c32cc269818a74e53909e044c4bd12f21b46d",
        inputs=WORKFLOW_INPUTS_0EFFA6A5,
        gates=WORKFLOW_GATE_ORDER_6F11BE26,
    ),
}


@dataclass(frozen=True, slots=True)
class CallerConfig:
    path: Path
    workflow_sha: str
    refs: dict[str, str]
    validate_roots: str
    run_generated_guard: bool
    guard_programs_root: bool
    release_base_url: str = DEFAULT_RELEASE_BASE_URL
    run_pytest: bool = True
    run_money_atom_check: bool = True
    # Every workflow_call input the pin declares, after defaults and
    # expression resolution, keyed by input name.
    inputs: Mapping[str, Any] = field(default_factory=dict)
    # How each caller expression was resolved locally, for the report.
    resolutions: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class CallerOverrides:
    """Local values for caller expressions that CI resolves from GitHub."""

    registry_url: str | None = None
    registry_anon_key: str | None = None


@dataclass(frozen=True, slots=True)
class GateSpec:
    key: str
    name: str
    subcommand: str
    flags: tuple[str, ...]
    workflow_lines: str


# Stable order is part of the public parity contract.  Dynamic file operands
# are represented by placeholders; literal flags match the workflow run blocks.
CI_GATE_REGISTRY: tuple[GateSpec, ...] = (
    GateSpec(
        "repository_tests", "Run repository tests", "pytest", ("-q", "tests"), "646-653"
    ),
    GateSpec(
        "obsolete_files",
        "Reject obsolete generated files",
        "find",
        ("*.rac", "*.rac.test"),
        "655-665",
    ),
    GateSpec(
        "repository_layout",
        "Reject disallowed repository layout",
        "layout",
        (),
        "667-835",
    ),
    GateSpec(
        "validation_waivers",
        "Enforce validation waiver ratchet",
        "validation-waivers audit",
        (
            "--root",
            "{repo}",
            "--corpus-path",
            "{corpus}",
            "--protected-base",
            "{base-waivers}",
            "--changed-paths",
            "{changed-paths}",
            "--axiom-rules-engine-path",
            "{engine}",
        ),
        "837-908",
    ),
    GateSpec(
        "guard_generated",
        "Reject manual RuleSpec changes",
        "guard-generated",
        (
            "--repo",
            "{repo}",
            "--base-ref",
            "{base-ref}",
            "--head-ref",
            "HEAD",
            "--corpus-path",
            "{corpus}",
            "--expected-encoder-checkout",
            "{encode}",
        ),
        "910-935",
    ),
    GateSpec(
        "select_targets",
        "Select RuleSpec validation targets",
        "selection",
        ("--base-ref", "{base-ref}", "--roots", "{roots}"),
        "936-1108",
    ),
    GateSpec(
        "validate",
        "Validate RuleSpec YAML",
        "validate",
        (
            "{files}",
            "--skip-reviewers",
            "--corpus-path",
            "{corpus}",
            "--axiom-rules-engine-path",
            "{engine}",
        ),
        "1109-1166",
    ),
    GateSpec(
        "companion_tests",
        "Execute RuleSpec companion tests",
        "test",
        (
            "--root",
            "{jurisdiction}",
            "--axiom-rules-engine-path",
            "{engine}",
            "{tests}",
        ),
        "1167-1218",
    ),
    GateSpec(
        "proof_validate",
        "Validate RuleSpec proofs and claims",
        "proof-validate",
        ("{files}", "--corpus-path", "{corpus}"),
        "1219-1249",
    ),
    GateSpec(
        "money_atoms",
        "Require money proof atoms",
        "proof-validate",
        ("{all-files}", "--money-atoms-only", "--corpus-path", "{corpus}", "{ratchet}"),
        "1250-1302",
    ),
    GateSpec(
        "oracle_coverage",
        "Validate PolicyEngine oracle coverage classification",
        "oracle-coverage",
        (
            "--root",
            "{repo}",
            "--fail-on-unmapped",
            "--fail-on-untested-comparable",
            "--limit",
            "50",
        ),
        "1303-1313",
    ),
    GateSpec(
        "changed_oracle_coverage",
        "Validate changed PolicyEngine oracle coverage classification",
        "oracle-coverage",
        ("--root", "{repo}", "--json"),
        "1314-1419",
    ),
)

# Gate names, subcommands and literal flags for the pins that execute the
# workflow's own scripts. Placeholders stand for dynamic operands, as in
# CI_GATE_REGISTRY; workflow line ranges are per pinned workflow file.
_WORKFLOW_GATE_DEFINITIONS: dict[str, tuple[str, str, tuple[str, ...]]] = {
    "unsupported_paths": (
        "Reject unsupported tracked paths",
        "tracked-paths",
        ("{tracked}", "{changed-paths}"),
    ),
    "unmanifested_rulespec": (
        "Reject unmanifested RuleSpec content",
        "manifest-precheck",
        ("{base-sha}...{head-sha}",),
    ),
    "migration_authorization": (
        "Authorize exact reviewed migration",
        "migration-authorization",
        ("{migration-authorization-path}",),
    ),
    "retired_schema_freeze": (
        "Verify immutable retired-schema freeze",
        "retired-schema-freeze",
        (".axiom/retired-schema-freeze.json",),
    ),
    "obsolete_files": (
        "Reject obsolete generated files",
        "find",
        ("*.rac", "*.rac.test"),
    ),
    "repository_layout": ("Reject disallowed repository layout", "layout", ()),
    "validation_waivers": (
        "Enforce validation waiver ratchet",
        "validation-waivers audit",
        (
            "--root",
            "{repo}",
            "--corpus-path",
            "{corpus}",
            "--protected-base",
            "{base-waivers}",
            "--changed-paths",
            "{audit-changed-paths}",
            "--partition-key",
            "{shard}",
            "--partition-keys-json",
            "{matrix}",
            "--axiom-rules-engine-path",
            "{engine}",
        ),
    ),
    "guard_generated": (
        "Reject manual RuleSpec changes",
        "guard-generated",
        (
            "--repo",
            "{repo}",
            "--base-ref",
            "{base-sha}",
            "--head-ref",
            "{head-sha}",
            "--corpus-path",
            "{corpus}",
            "--expected-encoder-checkout",
            "{encode}",
        ),
    ),
    "select_targets": (
        "Select RuleSpec validation targets",
        "selection",
        ("--base-ref", "{base-sha}", "--roots", "{shard-roots}"),
    ),
    "validate": (
        "Validate RuleSpec YAML",
        "validate",
        (
            "{files}",
            "--skip-reviewers",
            "--corpus-path",
            "{corpus}",
            "--axiom-rules-engine-path",
            "{engine}",
        ),
    ),
    "companion_tests": (
        "Execute RuleSpec companion tests",
        "test",
        (
            "--root",
            "{jurisdiction}",
            "--axiom-rules-engine-path",
            "{engine}",
            "{tests}",
        ),
    ),
    "proof_validate": (
        "Validate RuleSpec proofs and claims",
        "proof-validate",
        ("{files}", "--corpus-path", "{corpus}"),
    ),
    "money_atoms": (
        "Require money proof atoms",
        "proof-validate",
        ("{all-files}", "--money-atoms-only", "--corpus-path", "{corpus}", "{ratchet}"),
    ),
    "oracle_coverage": (
        "Validate PolicyEngine oracle coverage classification",
        "oracle-coverage",
        (
            "--root",
            "{repo}",
            "--fail-on-unmapped",
            "--fail-on-untested-comparable",
            "--limit",
            "50",
        ),
    ),
    "changed_oracle_coverage": (
        "Validate changed PolicyEngine oracle coverage classification",
        "oracle-coverage",
        ("--root", "{repo}", "--json"),
    ),
    "repository_tests": ("Run repository tests", "pytest", ("-q", "tests")),
}


def _workflow_gate_registry(
    order: Sequence[str], lines: Mapping[str, str]
) -> tuple[GateSpec, ...]:
    return tuple(
        GateSpec(key, *_WORKFLOW_GATE_DEFINITIONS[key], lines[key]) for key in order
    )


WORKFLOW_GATE_REGISTRY_0EFFA6A5 = _workflow_gate_registry(
    WORKFLOW_GATE_ORDER_0EFFA6A5,
    {
        "unsupported_paths": "155-200",
        "migration_authorization": "380-530",
        "retired_schema_freeze": "783-996",
        "obsolete_files": "1154-1164",
        "repository_layout": "1166-1334",
        "validation_waivers": "1336-1522",
        "guard_generated": "1524-1548",
        "select_targets": "1550-1739",
        "validate": "1741-1856",
        "companion_tests": "1858-1908",
        "proof_validate": "1910-1975",
        "money_atoms": "1977-2044",
        "oracle_coverage": "2046-2055",
        "changed_oracle_coverage": "2057-2161",
        "repository_tests": "2165-2172",
    },
)
WORKFLOW_GATE_REGISTRY_6F11BE26 = _workflow_gate_registry(
    WORKFLOW_GATE_ORDER_6F11BE26,
    {
        "unsupported_paths": "155-200",
        "unmanifested_rulespec": "202-553",
        "migration_authorization": "733-883",
        "retired_schema_freeze": "1136-1349",
        "obsolete_files": "1507-1517",
        "repository_layout": "1519-1687",
        "validation_waivers": "1689-1968",
        "guard_generated": "1970-2004",
        "select_targets": "2006-2195",
        "validate": "2197-2312",
        "companion_tests": "2314-2364",
        "proof_validate": "2366-2431",
        "money_atoms": "2433-2500",
        "oracle_coverage": "2502-2511",
        "changed_oracle_coverage": "2513-2620",
        "repository_tests": "2621-2628",
    },
)


def gate_registry_for_pin(workflow_sha: str) -> tuple[GateSpec, ...]:
    """Return the ordered gate registry the pinned workflow runs."""

    if workflow_sha == "0effa6a5b05e7fac53902df7d523e909bd7fc48a":
        return WORKFLOW_GATE_REGISTRY_0EFFA6A5
    if workflow_sha == "6f11be2655f79dd0a3b582db46525f58332ca120":
        return WORKFLOW_GATE_REGISTRY_6F11BE26
    if workflow_sha in SUPPORTED_WORKFLOW_PINS:
        return CI_GATE_REGISTRY
    raise KeyError(workflow_sha)


WORKFLOW_GATE_STEPS: dict[str, tuple[str, ...]] = {
    "Run repository tests": ("repository_tests",),
    "Reject obsolete generated files": ("obsolete_files",),
    "Reject disallowed repository layout": ("repository_layout",),
    "Enforce validation waiver ratchet": ("validation_waivers",),
    "Reject manual RuleSpec changes": ("guard_generated",),
    "Select RuleSpec validation targets": ("select_targets",),
    "Validate RuleSpec YAML": ("validate",),
    "Execute RuleSpec companion tests": ("companion_tests",),
    "Validate RuleSpec proofs and claims": ("proof_validate",),
    "Require money proof atoms": ("money_atoms",),
    "Validate PolicyEngine oracle coverage classification": ("oracle_coverage",),
    "Checkout changed-file oracle coverage classifier": ("changed_oracle_coverage",),
    "Install changed-file oracle coverage classifier": ("changed_oracle_coverage",),
    "Validate changed PolicyEngine oracle coverage classification": (
        "changed_oracle_coverage",
    ),
    "Reject unsupported tracked paths": ("unsupported_paths",),
    "Reject unmanifested RuleSpec content": ("unmanifested_rulespec",),
    "Authorize exact reviewed migration": ("migration_authorization",),
    "Verify immutable retired-schema freeze": ("retired_schema_freeze",),
}

# Every other step of a pinned workflow, with how ci accounts for it.
# `resolution` steps are reproduced before any gate runs; a failure there is a
# resolution failure. `environment` steps provision what the local checkouts,
# the ambient interpreter and the explicit keyring already provide.
WORKFLOW_RESOLUTION_STEPS: dict[str, str] = {
    "Compute validation shards": "compute_shard_plan",
    "Resolve shard validation roots": "ShardPlan.shard_roots",
    "Resolve RuleSpec toolchain": "resolve_workflow_toolchain",
    "Validate immutable dependency inputs": "verify_dependency_inputs",
    "Authenticate dependency commits": "verify_dependency_checkout",
    "Fetch pinned signed corpus release object": "acquire_release_object",
    "Authenticate signed corpus provenance commit": ("authenticate_release_provenance"),
    "Provision protected verification supervisor": (
        "local_corpus_release_verification keyring"
    ),
    "Check validation matrix result": "ci_verdict",
}
WORKFLOW_ENVIRONMENT_STEPS: dict[str, str] = {
    "Checkout rules repository": "the --repo checkout",
    "Free runner disk space": "runner housekeeping",
    "Set up Python": "the ambient interpreter",
    "Checkout axiom-encode": "the --encode-path checkout",
    "Checkout axiom-rules-engine": "the --engine-path checkout",
    "Checkout axiom-corpus": "the --corpus-path checkout",
    "Checkout rulespec-us canonical targets": "the --rulespec-us-path checkout",
    "Install Rust toolchain": "the local engine build",
    "Set up Go 1.26.1": "the supervisor build, replaced by library verification",
    "Install Python dependencies": "the ambient encoder environment",
    "Build axiom-rules-engine": "the local engine build",
    "Expose axiom-rules-engine on PATH": "the encoder resolves the engine path",
    "Link axiom-rules-engine as sibling checkout": "explicit checkout paths",
}


@dataclass(slots=True)
class GateResult:
    gate: str
    name: str
    status: str
    command: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    output: str = ""
    note: str | None = None


@dataclass(frozen=True, slots=True)
class DependencyMismatch:
    name: str
    head_sha: str
    pinned_sha: str

    def banner_line(self) -> str:
        return f"{self.name}: HEAD {self.head_sha} != pinned {self.pinned_sha}"


@dataclass(frozen=True, slots=True)
class Selection:
    mode: str
    rulespec_files: tuple[Path, ...]
    test_files: tuple[Path, ...]


def register_ci_parser(subparsers: Any) -> None:
    parser = subparsers.add_parser(
        "ci", help="Run the local validate-rulespec CI parity gate sequence"
    )
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--corpus-path", type=Path)
    parser.add_argument("--engine-path", type=Path)
    parser.add_argument("--rulespec-us-path", type=Path)
    parser.add_argument("--encode-path", type=Path)
    parser.add_argument("--base-ref", default="origin/main")
    parser.add_argument("--roots", default=None)
    parser.add_argument("--corpus-release-public-key", required=True)
    parser.add_argument(
        "--corpus-release-retired-public-key",
        action="append",
        default=[],
        help=(
            "Retired corpus release public key that workflow pins from "
            "0effa6a5 on provision beside the current key (repeatable)"
        ),
    )
    parser.add_argument(
        "--apply-public-key",
        help=(
            "Apply-manifest root the supervisor is provisioned with, to verify "
            "signed encoder apply manifests (AXIOM_ENCODE_APPLY_SIGNING_PUBLIC_KEY)"
        ),
    )
    parser.add_argument(
        "--eval-public-key",
        help=(
            "Eval-evidence root the supervisor is provisioned with "
            "(AXIOM_ENCODE_EVAL_SIGNING_PUBLIC_KEY)"
        ),
    )
    parser.add_argument(
        "--corpus-release-registry-url",
        help=(
            "Value for a caller corpus-release-registry-url passed as "
            f"${{{{ vars.* }}}} (default: {DEFAULT_RELEASE_REGISTRY_URL})"
        ),
    )
    parser.add_argument(
        "--corpus-release-registry-anon-key",
        help=(
            "Value for a caller corpus-release-registry-anon-key passed as "
            f"${{{{ vars.* }}}} (default: ${RELEASE_REGISTRY_ANON_KEY_ENV})"
        ),
    )
    parser.add_argument(
        "--pull-request",
        type=int,
        help=(
            "Pull-request number, needed only to reproduce a reviewed-migration "
            "authorization"
        ),
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help=(
            "Run independent supervised invocations (waiver-audit partitions, "
            "validation chunks, companion-test groups) in up to this many "
            "isolated processes; the default runs them in-process one at a time"
        ),
    )
    parser.add_argument("--allow-ref-mismatch", action="store_true")
    parser.add_argument("--allow-encoder-mismatch", action="store_true")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--json", action="store_true")


def _unsupported_pin_error(workflow_sha: str, path: Path) -> ValueError:
    supported = ", ".join(SUPPORTED_WORKFLOW_PINS)
    return ValueError(
        f"Unsupported validate-rulespec workflow pin {workflow_sha} "
        f"in {path}; this axiom-encode release implements "
        f"these pins: {supported}"
    )


class _WorkflowYamlLoader(yaml.SafeLoader):
    """SafeLoader resolving only YAML 1.2 core booleans (true/false).

    PyYAML follows YAML 1.1, where yes/no/on/off are booleans; GitHub reads a
    workflow's top-level ``on:`` as a string key, so those words are not
    booleans there. Leaving them strings makes a caller that passes one to a
    boolean input fail closed instead of silently passing.
    """


_WorkflowYamlLoader.yaml_implicit_resolvers = {
    first: [
        (tag, pattern) for tag, pattern in resolvers if tag != "tag:yaml.org,2002:bool"
    ]
    for first, resolvers in yaml.SafeLoader.yaml_implicit_resolvers.items()
}
_WorkflowYamlLoader.add_implicit_resolver(
    "tag:yaml.org,2002:bool",
    re.compile(r"^(?:true|True|TRUE|false|False|FALSE)$"),
    list("tTfF"),
)


def _caller_payload(path: Path) -> dict[str, Any]:
    try:
        payload = (
            yaml.load(path.read_text(encoding="utf-8"), Loader=_WorkflowYamlLoader)
            or {}
        )
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid caller workflow {path}: {exc}") from exc
    return payload if isinstance(payload, dict) else {}


def _caller_jobs(path: Path) -> dict[str, Any]:
    return _payload_jobs(_caller_payload(path))


def _payload_jobs(payload: Mapping[str, Any]) -> dict[str, Any]:
    jobs = payload.get("jobs", {})
    return jobs if isinstance(jobs, dict) else {}


def _validate_rulespec_jobs(
    path: Path, jobs: Mapping[str, Any]
) -> list[tuple[str, dict[str, Any]]]:
    found = []
    for name, job in jobs.items():
        if not isinstance(job, dict) or not isinstance(job.get("uses"), str):
            continue
        if job["uses"].endswith("validate-rulespec.yml@<pin-me>"):
            raise ValueError(
                f"Caller {path}: validate-rulespec workflow has placeholder pin "
                "<pin-me>; replace it with the reviewed full lowercase SHA"
            )
        match = WORKFLOW_RE.fullmatch(job["uses"])
        if match:
            found.append((match.group("sha"), job))
    return found


def find_caller_workflow(
    repo: Path, overrides: CallerOverrides | None = None
) -> CallerConfig:
    workflow_dir = repo / ".github" / "workflows"
    matches: list[tuple[Path, str, dict[str, Any], dict[str, Any]]] = []
    for path in sorted((*workflow_dir.glob("*.yml"), *workflow_dir.glob("*.yaml"))):
        payload = _caller_payload(path)
        jobs = _payload_jobs(payload)
        for workflow_sha, job in _validate_rulespec_jobs(path, jobs):
            matches.append((path, workflow_sha, job, payload))
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one validate-rulespec caller under {workflow_dir}; "
            f"found {len(matches)}"
        )
    path, workflow_sha, job, payload = matches[0]
    return parse_caller_job(
        path,
        workflow_sha,
        job,
        jobs=_payload_jobs(payload),
        repo=repo,
        overrides=overrides,
        workflow=payload,
    )


def parse_caller_workflow(
    path: Path, repo: Path | None = None, overrides: CallerOverrides | None = None
) -> CallerConfig:
    """Parse a caller fixture containing exactly one reusable-workflow job.

    ``repo`` is the checkout whose workflow-toolchain file resolves
    ``needs.<job>.outputs.*`` refs; it defaults to the checkout containing a
    ``.github/workflows/`` caller.
    """

    payload = _caller_payload(path)
    jobs = _payload_jobs(payload)
    found = _validate_rulespec_jobs(path, jobs)
    if len(found) != 1:
        raise ValueError(f"Expected one validate-rulespec caller in {path}")
    workflow_sha, job = found[0]
    if repo is None and path.parent.name == "workflows":
        repo = path.parents[2]
    return parse_caller_job(
        path,
        workflow_sha,
        job,
        jobs=jobs,
        repo=repo,
        overrides=overrides,
        workflow=payload,
    )


def parse_caller_job(
    path: Path,
    workflow_sha: str,
    job: dict[str, Any],
    *,
    jobs: Mapping[str, Any] | None = None,
    repo: Path | None = None,
    overrides: CallerOverrides | None = None,
    workflow: Mapping[str, Any] | None = None,
) -> CallerConfig:
    pin = SUPPORTED_WORKFLOW_PINS.get(workflow_sha)
    if pin is None:
        raise _unsupported_pin_error(workflow_sha, path)
    raw_inputs = job.get("with")
    if not isinstance(raw_inputs, dict):
        raise ValueError(f"Caller {path} has no with-inputs")
    unknown = sorted(str(name) for name in set(raw_inputs) - set(pin.inputs))
    if unknown:
        raise ValueError(
            f"Caller {path}: validate-rulespec@{workflow_sha[:8]} declares no "
            f"input {', '.join(unknown)}"
        )
    context = _CallerExpressionContext(
        path, job, jobs or {}, repo, overrides or CallerOverrides(), workflow or {}
    )
    inputs: dict[str, Any] = {}
    for name, declaration in pin.inputs.items():
        if name in raw_inputs:
            inputs[name] = context.resolve(name, raw_inputs[name], declaration)
        elif declaration.required:
            raise ValueError(f"Caller {path}: required input {name} is missing")
        elif declaration.default is None:
            inputs[name] = _INPUT_TYPE_DEFAULTS[declaration.type]
        else:
            inputs[name] = declaration.default
    # CI runs every job the validate-rulespec job needs before it, whether or
    # not an input reads its outputs; a failing one fails the run.
    context.run_needed_jobs()
    refs: dict[str, str] = {}
    for dependency, input_name in DEPENDENCY_INPUTS.items():
        value = inputs.get(input_name)
        if not isinstance(value, str) or SHA_RE.fullmatch(value) is None:
            raise ValueError(
                f"Caller {path}: {input_name} must be a full lowercase SHA"
            )
        refs[dependency] = value
    return CallerConfig(
        path=path,
        workflow_sha=workflow_sha,
        refs=refs,
        validate_roots=str(inputs["validate-roots"]),
        run_generated_guard=bool(inputs["run-generated-guard"]),
        guard_programs_root=bool(inputs["guard-programs-root"]),
        release_base_url=str(inputs["corpus-release-base-url"]),
        run_pytest=bool(inputs["run-pytest"]),
        run_money_atom_check=bool(inputs["run-money-atom-check"]),
        inputs=inputs,
        resolutions=tuple(context.resolutions),
    )


class _CallerExpressionContext:
    """Resolve the GitHub expressions a caller may pass as workflow inputs.

    Only whole-value expressions of two forms are supported:
    ``needs.<job>.outputs.<name>`` from a recognized workflow-toolchain resolver
    job, and ``vars.<NAME>`` for the release-registry inputs. Anything else
    fails closed rather than guessing at a value CI would compute.
    """

    def __init__(
        self,
        path: Path,
        job: Mapping[str, Any],
        jobs: Mapping[str, Any],
        repo: Path | None,
        overrides: CallerOverrides,
        workflow: Mapping[str, Any],
    ) -> None:
        self.path = path
        self.job = job
        self.jobs = jobs
        self.workflow = workflow
        self.repo = repo
        self.overrides = overrides
        self.resolutions: list[str] = []
        self._resolver_outputs: dict[str, dict[str, str]] = {}

    def resolve(self, name: str, value: Any, declaration: WorkflowInput) -> Any:
        if isinstance(value, str) and "${{" in value:
            match = EXPRESSION_RE.fullmatch(value)
            if match is None or declaration.type != "string":
                raise ValueError(
                    f"Caller {self.path}: input {name} uses an expression ci "
                    f"cannot reproduce: {value!r}"
                )
            expression = match.group("expression")
            needs = NEEDS_OUTPUT_RE.fullmatch(expression)
            if needs is not None:
                resolved = self._needs_output(
                    name, needs.group("job"), needs.group("output")
                )
                self.resolutions.append(f"{name}: {value} -> {resolved}")
                return resolved
            variable = VARS_RE.fullmatch(expression)
            if variable is not None:
                return self._variable(name, value, variable.group("name"))
            raise ValueError(
                f"Caller {self.path}: input {name} uses an expression ci "
                f"cannot reproduce: {value!r}"
            )
        expected = {"string": str, "boolean": bool, "number": (int, float)}[
            declaration.type
        ]
        if not isinstance(value, expected) or (
            declaration.type != "boolean" and isinstance(value, bool)
        ):
            raise ValueError(
                f"Caller {self.path}: input {name} must be a {declaration.type}, "
                f"got {value!r}"
            )
        return value

    def _needed_jobs(self) -> list[str]:
        needs = self.job.get("needs", [])
        needs = [needs] if isinstance(needs, str) else needs
        if not isinstance(needs, list) or not all(
            isinstance(name, str) for name in needs
        ):
            raise ValueError(
                f"Caller {self.path}: the validate-rulespec job's needs is malformed"
            )
        return needs

    def _outputs_of(self, job_name: str) -> dict[str, str]:
        if job_name not in self._resolver_outputs:
            self._resolver_outputs[job_name] = _workflow_toolchain_outputs(
                self.path, self.jobs, job_name, self.repo, workflow=self.workflow
            )
        return self._resolver_outputs[job_name]

    def run_needed_jobs(self) -> None:
        for job_name in self._needed_jobs():
            self._outputs_of(job_name)

    def _needs_output(self, input_name: str, job_name: str, output: str) -> str:
        if job_name not in self._needed_jobs():
            raise ValueError(
                f"Caller {self.path}: {input_name} reads needs.{job_name}, but "
                f"the validate-rulespec job does not list {job_name} in needs"
            )
        outputs = self._outputs_of(job_name)
        if output not in outputs:
            raise ValueError(
                f"Caller {self.path}: job {job_name} has no output {output}"
            )
        return outputs[output]

    def _variable(self, input_name: str, expression: str, variable: str) -> str:
        if input_name == "corpus-release-registry-url":
            if self.overrides.registry_url is not None:
                resolved = self.overrides.registry_url
                source = "--corpus-release-registry-url"
            elif variable == RELEASE_REGISTRY_URL_VARIABLE:
                resolved = DEFAULT_RELEASE_REGISTRY_URL
                source = "the organization registry default"
            else:
                raise ValueError(
                    f"Caller {self.path}: {input_name} comes from {expression}, "
                    "which a local run cannot read; pass "
                    "--corpus-release-registry-url (gh api "
                    f"repos/<owner>/<repo>/actions/variables/{variable} --jq .value)"
                )
            self.resolutions.append(
                f"{input_name}: {expression} -> {resolved} ({source})"
            )
            return resolved
        if input_name == "corpus-release-registry-anon-key":
            if self.overrides.registry_anon_key:
                source = "--corpus-release-registry-anon-key"
                resolved = self.overrides.registry_anon_key
            elif variable == RELEASE_REGISTRY_ANON_KEY_ENV and os.environ.get(
                RELEASE_REGISTRY_ANON_KEY_ENV
            ):
                source = f"${RELEASE_REGISTRY_ANON_KEY_ENV}"
                resolved = os.environ[RELEASE_REGISTRY_ANON_KEY_ENV]
            else:
                hint = (
                    f" or set {RELEASE_REGISTRY_ANON_KEY_ENV}"
                    if variable == RELEASE_REGISTRY_ANON_KEY_ENV
                    else ""
                )
                raise ValueError(
                    f"Caller {self.path}: {input_name} comes from {expression}, "
                    "which a local run cannot read; pass "
                    f"--corpus-release-registry-anon-key{hint} (gh api "
                    f"repos/<owner>/<repo>/actions/variables/{variable} --jq .value)"
                )
            self.resolutions.append(f"{input_name}: {expression} -> <{source}>")
            return resolved
        raise ValueError(
            f"Caller {self.path}: input {input_name} reads {expression}; ci "
            "resolves repository variables only for the corpus release registry "
            "inputs"
        )


def _workflow_toolchain_outputs(
    path: Path,
    jobs: Mapping[str, Any],
    job_name: str,
    repo: Path | None,
    *,
    workflow: Mapping[str, Any] | None = None,
) -> dict[str, str]:
    """Run a recognized workflow-toolchain resolver job's script locally."""

    def unrecognized(reason: str) -> ValueError:
        return ValueError(
            f"Caller {path}: job {job_name} is not a recognized workflow-toolchain "
            f"resolver ({reason}); ci resolves needs.* refs only through a "
            "reviewed resolver"
        )

    job = jobs.get(job_name)
    if not isinstance(job, dict) or "uses" in job:
        raise unrecognized("no such job")
    # Workflow-level env and defaults (e.g. defaults.run.working-directory)
    # reach every job's run steps, so the reviewed script would run differently.
    inherited = sorted(key for key in ("env", "defaults") if (workflow or {}).get(key))
    if inherited:
        raise unrecognized(f"the caller workflow sets {', '.join(inherited)}")
    extra_keys = set(job) - {"name", "runs-on", "outputs", "steps", "permissions"}
    if extra_keys:
        raise unrecognized(f"unexpected job keys {sorted(extra_keys)}")
    steps = job.get("steps")
    if not isinstance(steps, list) or len(steps) != 2:
        raise unrecognized("expected a checkout step and one resolver step")
    checkout, resolver = steps
    if (
        not isinstance(checkout, dict)
        or set(checkout) - {"name", "uses"}
        or not str(checkout.get("uses", "")).startswith("actions/checkout@")
    ):
        raise unrecognized("the first step is not a plain rules-repository checkout")
    if (
        not isinstance(resolver, dict)
        or set(resolver) - {"name", "id", "shell", "run"}
        or resolver.get("shell") != "bash"
        or not isinstance(resolver.get("id"), str)
        or not isinstance(resolver.get("run"), str)
    ):
        raise unrecognized("the resolver step has an unexpected shape")
    digest = hashlib.sha256(resolver["run"].encode("utf-8")).hexdigest()
    if digest not in RECOGNIZED_WORKFLOW_TOOLCHAIN_RESOLVERS:
        raise unrecognized(f"resolver script sha256 {digest} is not reviewed")
    declared = job.get("outputs")
    if not isinstance(declared, dict):
        raise unrecognized("no outputs")
    if repo is None:
        raise unrecognized("no checkout to read its workflow toolchain from")
    code, stdout, stderr = _run_embedded_python(
        _heredoc_bodies(resolver["run"])[0],
        cwd=repo,
        environment={"GITHUB_OUTPUT": os.devnull},
    )
    if code:
        raise ValueError(
            f"Caller {path}: workflow-toolchain job {job_name} fails: "
            f"{(stderr or stdout).strip()}"
        )
    step_outputs = dict(
        line.split("=", 1) for line in stdout.splitlines() if "=" in line
    )
    outputs: dict[str, str] = {}
    for name, value in declared.items():
        match = EXPRESSION_RE.fullmatch(value) if isinstance(value, str) else None
        source = STEP_OUTPUT_RE.fullmatch(match.group("expression")) if match else None
        if (
            source is None
            or source.group("step") != resolver["id"]
            or source.group("output") not in step_outputs
        ):
            raise unrecognized(f"output {name} is not a resolver step output")
        outputs[str(name)] = step_outputs[source.group("output")]
    return outputs


def resolve_dependency_paths(args: argparse.Namespace, repo: Path) -> dict[str, Path]:
    running_checkout = Path(__file__).resolve().parents[2]
    return {
        "encode": (args.encode_path or running_checkout).expanduser().resolve(),
        "engine": (args.engine_path or repo.parent / "axiom-rules-engine")
        .expanduser()
        .resolve(),
        "corpus": (args.corpus_path or repo.parent / "axiom-corpus")
        .expanduser()
        .resolve(),
        "rulespec_us": (args.rulespec_us_path or repo.parent / "rulespec-us")
        .expanduser()
        .resolve(),
    }


def _git_environment() -> dict[str, str]:
    """The ambient environment minus GIT_* values (GIT_DIR, GIT_CONFIG_*...).

    CI's steps see no such values, and identity checks must not be steered by
    a hook's or a `git -c` caller's environment.
    """

    return {
        name: value for name, value in os.environ.items() if not name.startswith("GIT_")
    }


def _git(
    repo: Path, *arguments: str, check: bool = True
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(repo), *arguments],
        check=check,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=_git_environment(),
    )


def verify_dependency_checkout(
    name: str,
    path: Path,
    pin: str,
    caller: Path,
    *,
    allow_ref_mismatch: bool,
    ignored_untracked: frozenset[str] = frozenset(),
) -> DependencyMismatch | None:
    if not path.is_dir():
        raise ValueError(f"{name} checkout does not exist: {path}")
    if _git(path, "cat-file", "-e", f"{pin}^{{commit}}", check=False).returncode:
        raise ValueError(
            f"{name} checkout {path} does not contain pinned commit {pin} from {caller}"
        )
    protected = "refs/remotes/origin/main"
    if _git(path, "show-ref", "--verify", "--quiet", protected, check=False).returncode:
        raise ValueError(
            f"{name} checkout {path} has no origin/main for authenticating {pin} from {caller}"
        )
    if _git(
        path, "merge-base", "--is-ancestor", pin, protected, check=False
    ).returncode:
        raise ValueError(
            f"{name} pinned commit {pin} from {caller} is not an ancestor of origin/main in {path}"
        )
    head = _git(path, "rev-parse", "HEAD").stdout.strip()
    # A checkout at the pinned HEAD but with local modifications is not the
    # pinned tree CI will use; treat dirtiness as a mismatch in its own right.
    # The only exemption is ci's own cache of the pinned release object, which
    # CI also writes into its corpus checkout before any gate runs.
    status = _git(path, "status", "--porcelain", "--untracked-files=all", check=False)
    # A status that cannot be read is not evidence of a clean checkout, and
    # assume-unchanged or skip-worktree entries hide edits from it.
    dirty = (
        status.returncode != 0
        or any(
            line and not (line.startswith("?? ") and line[3:] in ignored_untracked)
            for line in status.stdout.splitlines()
        )
        or bool(_index_flags(path))
    )
    if head == pin and not dirty:
        return None
    described_head = f"{head} (dirty worktree)" if dirty else head
    mismatch = DependencyMismatch(name, described_head, pin)
    warning = f"REF MISMATCH: {mismatch.banner_line()} declared by {caller}"
    if not allow_ref_mismatch:
        raise ValueError(warning + "; pass --allow-ref-mismatch to continue")
    return mismatch


def verify_ambient_encoder(
    encode_pin: str,
    pinned_version: str,
    caller: Path,
    *,
    allow_encoder_mismatch: bool,
) -> DependencyMismatch | None:
    """Bind the imported encoder to the caller pin, preferring source identity."""

    source_checkout = _ambient_source_checkout()
    result = (
        _git(source_checkout, "rev-parse", "HEAD", check=False)
        if source_checkout is not None
        else None
    )
    head = result.stdout.strip() if result is not None else ""
    if result is not None and result.returncode == 0 and SHA_RE.fullmatch(head):
        # Modified or added sources at the pinned HEAD are not the pinned
        # encoder, whatever the checkout's status.showUntrackedFiles says; a
        # status that cannot be read is not evidence of a clean checkout.
        status = _git(
            source_checkout,
            "status",
            "--porcelain",
            "--untracked-files=all",
            "--",
            "src",
            "pyproject.toml",
            "uv.lock",
            check=False,
        )
        dirty = (
            status.returncode != 0
            or bool(status.stdout.strip())
            or bool(_index_flags(source_checkout, "src", "pyproject.toml", "uv.lock"))
        )
        mismatch = (
            None
            if head == encode_pin and not dirty
            else DependencyMismatch(
                "ambient-encoder",
                f"{head} (dirty worktree)" if dirty else head,
                encode_pin,
            )
        )
    else:
        # Version equality cannot establish source identity: an installed
        # wheel with no resolvable git HEAD may differ from the pinned
        # encoder at an equal version string. Unresolvable identity is
        # always a mismatch (fail-closed without the flag).
        mismatch = DependencyMismatch(
            "ambient-encoder",
            f"unresolvable ({__version__})",
            encode_pin,
        )
    if mismatch is not None and not allow_encoder_mismatch:
        raise ValueError(
            f"ENCODER MISMATCH: {mismatch.banner_line()} declared by {caller}; "
            "run this tool from the pinned encode checkout for exact parity, or "
            "pass --allow-encoder-mismatch for a qualified development verdict"
        )
    return mismatch


def _ambient_source_checkout() -> Path | None:
    """Return the checkout the imported package runs from, if it is one.

    Only an ``<checkout>/src/axiom_encode`` layout whose Git top level is that
    checkout counts; an installed copy (say, a wheel under a virtualenv that
    happens to sit inside some checkout) has no source identity.
    """

    package = Path(__file__).resolve().parent
    checkout = package.parents[1]
    if package.parent.name != "src" or not (checkout / "pyproject.toml").is_file():
        return None
    top = _git(checkout, "rev-parse", "--show-toplevel", check=False)
    if top.returncode or Path(top.stdout.strip()).resolve() != checkout:
        return None
    return checkout


@contextlib.contextmanager
def committed_checkout(
    source: Path, head: str | None = None, base: str | None = None
) -> Iterator[Path]:
    """Yield a fresh checkout of ``source``'s HEAD, as actions/checkout makes.

    CI validates the committed tree of a fresh clone, with no local Git
    configuration, attributes, hooks or sparse-checkout state. The checkout is
    therefore an independent transport clone of ``source`` (``--no-local``,
    so every object is re-hashed on receipt and none is shared or linked),
    made from an empty template under empty global and system configuration
    and attributes, named like the source checkout and removed afterwards.
    ``head`` and ``base`` are fetched by commit, so a detached HEAD or a
    remote-tracking base is available. Its origin is ``source``'s origin, as
    GitHub's checkout points origin at the repository itself. A tree whose
    paths would collide on this filesystem (case or Unicode normalization)
    fails closed, because CI's Linux checkout keeps every path.
    """

    if head is None:
        head = resolve_commit(source, "HEAD")
    parent = Path(tempfile.mkdtemp(prefix="axiom-ci-checkout-")).resolve()
    target = parent / source.name
    environment = {
        **_git_environment(),
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_ATTR_NOSYSTEM": "1",
        "GIT_TERMINAL_PROMPT": "0",
    }

    def git(*arguments: str) -> str:
        return subprocess.run(
            ["git", "-c", f"core.attributesFile={os.devnull}", *arguments],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=environment,
        ).stdout.decode("utf-8", "surrogateescape")

    try:
        git(
            "clone",
            "--quiet",
            "--no-local",
            "--no-checkout",
            "--template=",
            str(source),
            str(target),
        )
        git(
            "-C",
            str(target),
            "fetch",
            "--quiet",
            "--no-tags",
            str(source),
            *(commit for commit in (head, base) if commit),
        )
        origin = _git(source, "remote", "get-url", "origin", check=False)
        if origin.returncode == 0 and origin.stdout.strip():
            git("-C", str(target), "remote", "set-url", "origin", origin.stdout.strip())
        else:
            git("-C", str(target), "remote", "remove", "origin")
        collisions = _colliding_paths(
            git("-C", str(target), "ls-tree", "-r", "-z", "--name-only", head).split(
                "\0"
            ),
            parent,
        )
        if collisions:
            raise ValueError(
                "committed paths collide on this filesystem (case or Unicode "
                "normalization), so a local checkout cannot hold what CI's "
                "Linux checkout validates: " + "; ".join(collisions[:10])
            )
        git("-C", str(target), "checkout", "--quiet", "--detach", head)
        yield target
    finally:
        shutil.rmtree(parent, ignore_errors=True)


def _colliding_paths(paths: Sequence[str], probe_directory: Path) -> list[str]:
    """Return groups of tracked paths (and directories) this filesystem merges."""

    probe = probe_directory / "Axiom-CI-Case-Probe"
    probe.write_text("")
    try:
        case_insensitive = (probe_directory / "axiom-ci-case-probe").exists()
    finally:
        probe.unlink()
    composed = probe_directory / unicodedata.normalize("NFC", "axiom-ci-\u00e9")
    composed.write_text("")
    try:
        normalization_insensitive = (
            probe_directory / unicodedata.normalize("NFD", "axiom-ci-\u00e9")
        ).exists()
    finally:
        composed.unlink()
    if not case_insensitive and not normalization_insensitive:
        return []

    def fold(text: str) -> str:
        if normalization_insensitive:
            text = unicodedata.normalize("NFC", text)
        return text.casefold() if case_insensitive else text

    seen: dict[str, set[str]] = {}
    for path in filter(None, paths):
        parts = path.split("/")
        for index in range(1, len(parts) + 1):
            prefix = "/".join(parts[:index])
            seen.setdefault(fold(prefix), set()).add(prefix)
    return [" = ".join(sorted(group)) for group in seen.values() if len(group) > 1]


def resolve_commit(repo: Path, ref: str) -> str:
    """Resolve ``ref`` to a full commit SHA in ``repo`` (fails if it cannot)."""

    result = _git(repo, "rev-parse", "--verify", f"{ref}^{{commit}}", check=False)
    sha = result.stdout.strip()
    if result.returncode or SHA_RE.fullmatch(sha) is None:
        raise ValueError(f"{ref} does not name a commit in {repo}")
    return sha


def _index_flags(repo: Path, *pathspec: str) -> list[str]:
    """Return tracked paths whose index hides worktree changes from status.

    ``git ls-files -v`` marks assume-unchanged entries with a lowercase tag
    and skip-worktree entries with ``S``.
    """

    listing = _git(repo, "ls-files", "-v", "--", *pathspec, check=False)
    if listing.returncode:
        return ["<git ls-files failed>"]
    return [
        line[2:]
        for line in listing.stdout.splitlines()
        if len(line) > 2 and (line[0].islower() or line[0] == "S")
    ]


def uncommitted_changes_note(source: Path) -> str | None:
    """Describe local state of ``source`` that the committed checkout omits."""

    status = _git(source, "status", "--porcelain", "--untracked-files=all", check=False)
    flagged = _index_flags(source)
    if status.returncode == 0 and not status.stdout.strip() and not flagged:
        return None
    return (
        f"{source} has uncommitted, untracked or index-hidden changes; ci "
        "validated its committed HEAD, as CI would"
    )


def verify_python_version(
    requested: str, caller: Path, *, allow_encoder_mismatch: bool
) -> DependencyMismatch | None:
    """Bind the interpreter to the caller's python-version (setup-python).

    CI's interpreter also runs unoptimized, so assertions in the in-process
    gates must not be stripped by -O or PYTHONOPTIMIZE.
    """

    running = f"{sys.version_info.major}.{sys.version_info.minor}"
    if sys.flags.optimize:
        running = f"{running} -{'O' * sys.flags.optimize}"
    if running == requested:
        return None
    mismatch = DependencyMismatch("python", running, requested)
    if not allow_encoder_mismatch:
        raise ValueError(
            f"PYTHON MISMATCH: running {running}, but {caller} sets up Python "
            f"{requested}; run ci under Python {requested} for exact parity, or "
            "pass --allow-encoder-mismatch for a qualified development verdict"
        )
    return mismatch


def encoder_version_at_pin(path: Path, pin: str) -> str:
    """Require consistent encoder version metadata at the caller's exact pin."""

    def show(name: str) -> str:
        result = _git(path, "show", f"{pin}:{name}", check=False)
        if result.returncode:
            raise ValueError(f"Pinned encoder {pin} has no {name}")
        return result.stdout

    pyproject = tomllib.loads(show("pyproject.toml"))["project"]["version"]
    package_match = re.search(
        r'(?m)^__version__\s*=\s*"([^"]+)"',
        show("src/axiom_encode/__init__.py"),
    )
    lock_match = re.search(
        r'(?ms)^name = "axiom-encode"\nversion = "([^"]+)"', show("uv.lock")
    )
    package = package_match.group(1) if package_match else None
    lock = lock_match.group(1) if lock_match else None
    if not isinstance(pyproject, str) or pyproject != package or pyproject != lock:
        raise ValueError(
            f"Pinned encoder {pin} has inconsistent version metadata "
            f"(pyproject={pyproject}, package={package}, lock={lock})"
        )
    return pyproject


def acquire_release_object(
    toolchain: RuleSpecToolchain,
    corpus_path: Path,
    base_url: str,
    *,
    offline: bool,
    fetcher: Callable[[str], bytes] | None = None,
) -> Path:
    destination = (
        corpus_path
        / "releases"
        / toolchain.corpus_release
        / f"{toolchain.corpus_release_content_sha256}.json"
    )
    if destination.is_file():
        return destination
    if offline:
        raise ValueError(
            f"--offline requires pinned corpus release object: {destination}"
        )
    url = f"{base_url.rstrip('/')}/releases/{toolchain.corpus_release}/{toolchain.corpus_release_content_sha256}.json"
    fetch = fetcher or (lambda value: _fetch_https(value, {}))
    raw = fetch(url)
    try:
        payload = json.loads(raw)
        content = payload["content"]
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ValueError(
            f"Corpus release acquisition error: invalid JSON: {exc}"
        ) from exc
    actual = hashlib.sha256(
        json.dumps(
            content, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode()
    ).hexdigest()
    if payload.get("release") != toolchain.corpus_release:
        raise ValueError("Corpus release acquisition error: release name mismatch")
    if (
        payload.get("content_sha256") != actual
        or actual != toolchain.corpus_release_content_sha256
    ):
        raise ValueError(
            f"Corpus release acquisition error: content sha256 mismatch ({actual} != {toolchain.corpus_release_content_sha256})"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    # A unique temporary name: concurrent runs may share this corpus checkout.
    descriptor, name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
        temporary.replace(destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def authenticate_release_provenance(
    release_path: Path,
    corpus_path: Path,
    corpus_pin: str,
    caller: Path,
) -> str:
    try:
        payload = json.loads(release_path.read_text(encoding="utf-8"))
        commit = payload["content"]["git"]["commit"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ValueError(f"Signed corpus release provenance is missing: {exc}") from exc
    if not isinstance(commit, str) or SHA_RE.fullmatch(commit) is None:
        raise ValueError("Signed release provenance is not a full lowercase commit SHA")
    if _git(
        corpus_path, "cat-file", "-e", f"{commit}^{{commit}}", check=False
    ).returncode:
        raise ValueError(f"Signed corpus release provenance commit is absent: {commit}")
    if _git(
        corpus_path,
        "merge-base",
        "--is-ancestor",
        commit,
        "refs/remotes/origin/main",
        check=False,
    ).returncode:
        raise ValueError(
            f"Signed corpus release provenance {commit} is not an ancestor of corpus origin/main"
        )
    if _git(
        corpus_path, "merge-base", "--is-ancestor", commit, corpus_pin, check=False
    ).returncode:
        raise ValueError(
            f"Signed corpus release provenance {commit} is not contained in "
            f"caller corpus pin {corpus_pin} from {caller}"
        )
    return commit


def verify_toolchain_base_binding(repo: Path, base_ref: str) -> None:
    """Mirror the workflow's migrated-base removal guard."""

    result = _git(repo, "show", f"{base_ref}:.axiom/toolchain.toml", check=False)
    if result.returncode:
        return
    try:
        payload = tomllib.loads(result.stdout)
    except tomllib.TOMLDecodeError:
        return
    table = payload.get("toolchain")
    expected = {
        "axiom_corpus_release",
        "axiom_corpus_release_content_sha256",
        "validation_waiver_set_sha256",
    }
    if isinstance(table, dict) and set(table) == expected:
        path = repo / ".axiom" / "toolchain.toml"
        if not path.is_file() or path.is_symlink():
            raise ValueError(
                ".axiom/toolchain.toml cannot be removed once a base branch uses it"
            )


def _glob_directories(repo: Path) -> list[Path]:
    """The directories ``for dir in */`` visits on the runner, in its order.

    Bash sorts ``*/`` matches with their trailing slash under the runner's
    C.UTF-8 collation (code point order), so ``us-ak/`` precedes ``us/``
    ('-' < '/'); sorting bare names would put ``us`` first. rulespec-us's
    shards job lists us-ak first and us last (Repository Checks run
    36184228440), and the first shard hosts the repository-wide gates.
    """

    return sorted(
        (
            child
            for child in repo.iterdir()
            if not child.name.startswith(".") and child.is_dir()
        ),
        key=lambda child: f"{child.name}/",
    )


def resolve_roots(repo: Path, raw: str) -> tuple[str, ...]:
    if raw != "auto":
        return tuple(item for item in raw.split() if item)
    candidates = []
    for child in _glob_directories(repo):
        if re.fullmatch(r"[a-z]{2}(?:-[a-z0-9-]+)*", child.name) and any(
            (child / root).is_dir()
            for root in ("statutes", "regulations", "policies", "legislation")
        ):
            candidates.append(child.name)
    if not candidates:
        for name in ("statutes", "regulations", "policies", "legislation"):
            if (repo / name).is_dir():
                candidates.append(name)
    return tuple(candidates)


def _changed_paths(repo: Path, base_ref: str) -> tuple[str, ...]:
    result = _git(
        repo,
        "diff",
        "--name-only",
        "--no-renames",
        "--diff-filter=ACMRTD",
        base_ref,
        "HEAD",
    )
    return tuple(line for line in result.stdout.splitlines() if line)


def select_targets(repo: Path, base_ref: str, roots: Sequence[str]) -> Selection:
    changed = _changed_paths(repo, base_ref)
    mode = "changed"
    if any(path.startswith(".github/workflows/") for path in changed):
        mode = "full-toolchain-bump"
    elif ".axiom/toolchain.toml" in changed:
        before = _git(repo, "show", f"{base_ref}:.axiom/toolchain.toml", check=False)
        try:
            old = tomllib.loads(before.stdout).get("toolchain", {})
            new = tomllib.loads((repo / ".axiom/toolchain.toml").read_text()).get(
                "toolchain", {}
            )
            differences = {
                key for key in set(old) | set(new) if old.get(key) != new.get(key)
            }
        except (OSError, tomllib.TOMLDecodeError):
            differences = {"invalid"}
        if differences and differences <= {
            "axiom_corpus_release",
            "axiom_corpus_release_content_sha256",
        }:
            mode = "changed-corpus-toolchain-bump"
        else:
            mode = "full-toolchain-bump"
    rules: set[Path] = set()
    tests: set[Path] = set()

    def add(path_text: str) -> None:
        if "/programs/" in f"/{path_text}/":
            return
        path = repo / path_text
        if path_text.endswith(".test.yaml"):
            tests.add(path) if path.is_file() else None
            module = repo / f"{path_text[:-10]}.yaml"
            rules.add(module) if module.is_file() else None
        elif path_text.endswith(".test.yml"):
            tests.add(path) if path.is_file() else None
            module = repo / f"{path_text[:-9]}.yml"
            rules.add(module) if module.is_file() else None
        elif path_text.endswith((".yaml", ".yml")):
            rules.add(path) if path.is_file() else None
            suffix = ".test.yaml" if path_text.endswith(".yaml") else ".test.yml"
            companion = path.with_name(path.name.rsplit(".", 1)[0] + suffix)
            tests.add(companion) if companion.is_file() else None

    if mode == "full-toolchain-bump":
        for root in roots:
            if (repo / root).is_dir():
                for path in (repo / root).rglob("*.y*ml"):
                    add(path.relative_to(repo).as_posix())
    else:
        for path in changed:
            if any(path.startswith(f"{root}/") for root in roots):
                add(path)
    return Selection(mode, tuple(sorted(rules)), tuple(sorted(tests)))


def _run_cli(
    arguments: Sequence[str],
    *,
    environment: dict[str, str] | None = None,
    cwd: Path | None = None,
    supervised: bool = False,
) -> tuple[int, str]:
    """Run one encoder subcommand in-process and capture its output.

    ``supervised`` mirrors the protected supervisor's child environment: it is
    built from empty in CI, so AXIOM_* values the supervisor does not forward
    and every ambient GIT_* value (GIT_DIR, GIT_CONFIG_COUNT, ...) are
    withheld, and Git ignores global and system configuration. ``cwd``
    mirrors the step's working directory (the rules checkout in CI).
    """

    from axiom_encode import cli

    output = io.StringIO()
    old_argv = sys.argv
    old_environment = dict(os.environ)
    old_cwd = os.getcwd() if cwd is not None else None
    try:
        sys.argv = ["axiom-encode", *arguments]
        if supervised:
            for name in [
                name
                for name in os.environ
                if (
                    name.startswith("AXIOM_")
                    and name not in SUPERVISOR_FORWARDED_ENVIRONMENT
                )
                or name.startswith("GIT_")
            ]:
                del os.environ[name]
            os.environ.update(SUPERVISOR_GIT_ENVIRONMENT)
        if environment:
            os.environ.update(environment)
        if cwd is not None:
            os.chdir(cwd)
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
            try:
                cli.main()
            except SystemExit as exc:
                return _exit_status(exc.code, output), output.getvalue()
            except Exception as exc:  # keep running later gates for local iteration
                print(f"{type(exc).__name__}: {exc}", file=output)
                return 1, output.getvalue()
    finally:
        sys.argv = old_argv
        if old_cwd is not None:
            os.chdir(old_cwd)
        if dict(os.environ) != old_environment:
            os.environ.clear()
            os.environ.update(old_environment)
    return 0, output.getvalue()


def _exit_status(code: object, output: io.StringIO) -> int:
    """Map a SystemExit code the way the interpreter would for a process."""

    if code is None:
        return 0
    if isinstance(code, int):
        return code
    # sys.exit("message") prints the message and exits 1.
    print(code, file=output)
    return 1


def _run_pinned_cli(
    encode_path: Path, pin: str, arguments: Sequence[str]
) -> tuple[int, str]:
    """Execute classifier code from the caller's exact local encoder pin."""

    code, output, _ = _run_pinned_process(
        encode_path, pin, arguments, stderr=subprocess.STDOUT
    )
    return code, output


def _run_pinned_process(
    encode_path: Path, pin: str, arguments: Sequence[str], *, stderr: int
) -> tuple[int, str, str]:
    expected_oracles = _encoder_oracles_pin(encode_path, pin)
    installed_oracles = _installed_oracles_pin()
    if expected_oracles != installed_oracles:
        return (
            1,
            "Pinned changed-file classifier requires axiom-oracles "
            f"{expected_oracles}, but the local runtime has {installed_oracles or 'no VCS pin'}. "
            "Install the pinned encoder dependencies before running ci.\n",
            "",
        )

    with tempfile.TemporaryDirectory(prefix="axiom-ci-classifier-") as temp_name:
        archive = subprocess.run(
            ["git", "-C", str(encode_path), "archive", "--format=tar", pin],
            check=True,
            stdout=subprocess.PIPE,
            env=_git_environment(),
        ).stdout
        root = Path(temp_name)
        with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as bundle:
            bundle.extractall(root, filter="data")
        environment = dict(os.environ)
        existing = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = str(root / "src") + (
            os.pathsep + existing if existing else ""
        )
        result = subprocess.run(
            [sys.executable, "-m", "axiom_encode.cli", *arguments],
            cwd=root,
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=stderr,
            check=False,
        )
        return result.returncode, result.stdout, result.stderr or ""


def _encoder_oracles_pin(encode_path: Path, pin: str) -> str:
    payload = _git(encode_path, "show", f"{pin}:pyproject.toml").stdout
    match = re.search(
        r'axiom-oracles\s+@\s+git\+https://github\.com/[^"@]+@([0-9a-f]{40})',
        payload,
    )
    if match is None:
        raise ValueError(f"Pinned encoder {pin} has no exact axiom-oracles dependency")
    return match.group(1)


def _installed_oracles_pin() -> str | None:
    """Return the runtime's exact axiom-oracles VCS commit, never its version."""

    try:
        raw = importlib.metadata.distribution("axiom-oracles").read_text(
            "direct_url.json"
        )
        payload = json.loads(raw or "null")
        commit = payload.get("vcs_info", {}).get("commit_id")
    except (importlib.metadata.PackageNotFoundError, json.JSONDecodeError, TypeError):
        return None
    return commit if isinstance(commit, str) and SHA_RE.fullmatch(commit) else None


def _run_process(
    arguments: Sequence[str], cwd: Path, *, environment: Mapping[str, str] | None = None
) -> tuple[int, str]:
    result = subprocess.run(
        arguments,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        env=None if environment is None else dict(environment),
    )
    return result.returncode, result.stdout


def _repository_test_environment() -> dict[str, str]:
    """The environment of the workflow's `python -m pytest -q tests` step.

    CI supplies no PYTEST_* options (PYTEST_ADDOPTS could deselect failing
    tests), no PYTHON* interpreter overrides (PYTHONOPTIMIZE strips asserts)
    and no GIT_* steering, so none may reach the local run.
    """

    return {
        name: value
        for name, value in _git_environment().items()
        if not name.startswith(("PYTEST_", "PYTHON"))
    }


def _result(
    spec: GateSpec,
    code: int,
    output: str,
    command: Sequence[str],
    *,
    note: str | None = None,
) -> GateResult:
    failures = (
        [] if code == 0 else [line for line in output.splitlines() if line.strip()]
    )
    return GateResult(
        spec.key,
        spec.name,
        "PASS" if code == 0 else "FAIL",
        list(command),
        failures,
        output,
        note,
    )


def _obsolete_gate(repo: Path) -> tuple[int, str]:
    matches = [
        str(path.relative_to(repo))
        for path in repo.rglob("*")
        if path.is_file()
        and (path.name.endswith(".rac") or path.name.endswith(".rac.test"))
        and not any(
            part in {".git", "_axiom", ".venv", ".pytest_cache"}
            for part in path.relative_to(repo).parts
        )
    ]
    return (1, "\n".join(matches)) if matches else (0, "No obsolete generated files.\n")


def _layout_gate(repo: Path, roots: Sequence[str]) -> tuple[int, str]:
    config_path = repo / ".axiom" / "repository-structure.yaml"
    if config_path.is_file():
        config = yaml.safe_load(config_path.read_text()) or {}
        if not isinstance(config, dict):
            return 1, f"{config_path} must contain a mapping"
        if config.get("version") != 1:
            return 1, f"{config_path} must set version: 1"
        for key in ("allowed_root_directories", "allowed_root_files"):
            value = config.get(key)
            if not isinstance(value, list) or not all(
                isinstance(item, str) for item in value
            ):
                return 1, f"{key} must be a list of strings"
        declared_rules = config.get("path_rules")
        if not isinstance(declared_rules, list) or not declared_rules:
            return 1, "path_rules must be a non-empty list"
        for index, rule in enumerate(declared_rules, start=1):
            if not isinstance(rule, dict):
                return 1, f"path_rules[{index}] must be a mapping"
            patterns = rule.get("patterns")
            if (
                not isinstance(patterns, list)
                or not patterns
                or not all(isinstance(pattern, str) for pattern in patterns)
            ):
                return (
                    1,
                    f"path_rules[{index}].patterns must be a non-empty list of strings",
                )
            for key in ("allow_extensions", "allow_filenames"):
                value = rule.get(key, [])
                if not isinstance(value, list) or not all(
                    isinstance(item, str) for item in value
                ):
                    return 1, f"path_rules[{index}].{key} must be a list of strings"
        tracked = _git(repo, "ls-files", "-z").stdout.split("\0")
        problems = []
        root_dirs = {
            item.strip("/") for item in config.get("allowed_root_directories", [])
        }
        root_files = {item.strip("/") for item in config.get("allowed_root_files", [])}
        rules = config.get("path_rules", [])
        for path in filter(None, tracked):
            parts = PurePosixPath(path).parts
            if len(parts) == 1:
                if path not in root_files:
                    problems.append(f"{path}: top-level file is not allowed")
                continue
            if parts[0] not in root_dirs:
                problems.append(
                    f"{path}: top-level directory {parts[0]}/ is not allowed"
                )
                continue
            rule = next(
                (
                    item
                    for item in rules
                    if any(
                        fnmatch.fnmatchcase(path, pattern.strip("/"))
                        for pattern in item.get("patterns", [])
                    )
                ),
                None,
            )
            if rule is None:
                problems.append(f"{path}: no path rule matched")
            elif PurePosixPath(path).name not in rule.get(
                "allow_filenames", []
            ) and PurePosixPath(path).suffix not in rule.get("allow_extensions", []):
                problems.append(
                    f"{path}: file name/extension is not allowed by matched path rule"
                )
        return (
            (1, "\n".join(problems))
            if problems
            else (0, f"Repository layout matches {config_path}.\n")
        )
    problems = [
        name for name in ("statute", "regulation", "policy") if (repo / name).exists()
    ]
    for path in repo.rglob("*.y*ml"):
        relative = path.relative_to(repo)
        if any(
            part in {".git", "_axiom", ".venv", ".pytest_cache", ".github"}
            for part in relative.parts
        ):
            continue
        text = relative.as_posix()
        if path.name in {"parameters.yaml", "tests.yaml"} or (
            relative.parts and relative.parts[0] == "tests"
        ):
            problems.append(text)
            continue
        if path.name in {"known-dangling.yaml", "known-validation-gaps.yaml"}:
            continue
        if not any(text.startswith(f"{root}/") for root in roots):
            problems.append(text)
    return (
        (1, "\n".join(problems)) if problems else (0, "Repository layout is allowed.\n")
    )


def _classifier_gate(
    repo: Path,
    selection: Selection,
    encode_path: Path,
    encode_pin: str,
) -> tuple[int, str]:
    if not selection.rulespec_files:
        return 0, "No changed RuleSpec YAML files selected for oracle coverage.\n"
    code, output = _run_pinned_cli(
        encode_path,
        encode_pin,
        ["oracle-coverage", "--root", str(repo), "--json"],
    )
    if code:
        return code, output
    try:
        payload = json.loads(output)
    except json.JSONDecodeError as exc:
        return 1, f"Changed oracle coverage emitted invalid JSON: {exc}\n{output}"
    if repo.name.startswith("rulespec-be"):
        return 0, "Changed PolicyEngine oracle coverage skipped for Belgium.\n"
    changed = {
        f"{repo.name}/{path.relative_to(repo).as_posix()}"
        for path in selection.rulespec_files
    }
    failures = []
    items = [item for item in payload.get("items", []) if item.get("file") in changed]
    for item in items:
        if item.get("status") == "unmapped":
            failures.append(f"{item.get('legal_id')}: unmapped")
        elif item.get("status") == "comparable" and not item.get("tested"):
            failures.append(
                f"{item.get('legal_id')}: comparable but not covered by companion tests"
            )
    if failures:
        return 1, "Changed PolicyEngine oracle coverage is incomplete.\n" + "\n".join(
            f"- {failure}" for failure in failures
        )
    return (
        0,
        f"Changed PolicyEngine oracle coverage passed for {len(items)} output(s).\n",
    )


def execute_gates(
    args: argparse.Namespace,
    caller: CallerConfig,
    paths: dict[str, Path],
    roots: tuple[str, ...],
) -> list[GateResult]:
    repo = args.repo.resolve()
    selection = select_targets(repo, args.base_ref, roots)
    results: list[GateResult] = []
    specs = {spec.key: spec for spec in CI_GATE_REGISTRY}
    gate_parameters = SUPPORTED_WORKFLOW_PINS[caller.workflow_sha].gate_parameters
    if (
        caller.run_pytest
        and (repo / "tests").is_dir()
        and any(
            path.is_file()
            and (path.name.startswith("test_") or path.name.endswith("_test.py"))
            for path in (repo / "tests").rglob("*.py")
        )
    ):
        command = [sys.executable, "-m", "pytest", "-q", "tests"]
        results.append(
            _result(
                specs["repository_tests"],
                *_run_process(
                    command, repo, environment=_repository_test_environment()
                ),
                command,
            )
        )
    else:
        results.append(
            _result(
                specs["repository_tests"],
                0,
                "No Python tests found; skipping pytest.\n",
                [],
            )
        )
    results.append(
        _result(
            specs["obsolete_files"],
            *_obsolete_gate(repo),
            ["find", ".", "*.rac", "*.rac.test"],
        )
    )
    results.append(
        _result(
            specs["repository_layout"],
            # Workflow shard resolution always permits programs/: it is a
            # validation root when guarded, otherwise an allowed extra root.
            *_layout_gate(repo, (*roots, "sources", "programs")),
            ["layout"],
        )
    )
    with tempfile.TemporaryDirectory(prefix="axiom-ci-") as temp_name:
        temp = Path(temp_name)
        protected = temp / "protected-known-validation-gaps.yaml"
        changed = temp / "waiver-changed-paths.txt"
        base = _git(
            repo, "show", f"{args.base_ref}:known-validation-gaps.yaml", check=False
        )
        protected.write_text(base.stdout)
        changed.write_text("\n".join(_changed_paths(repo, args.base_ref)) + "\n")
        waiver_command = [
            "validation-waivers",
            "audit",
            "--root",
            str(repo),
            "--corpus-path",
            str(paths["corpus"]),
            "--protected-base",
            str(protected),
            "--changed-paths",
            str(changed),
            "--axiom-rules-engine-path",
            str(paths["engine"]),
        ]
        code, output = (
            (
                1,
                "ValidationWaiverBaseMissing: protected base must contain known-validation-gaps.yaml\n",
            )
            if base.returncode
            else _run_cli(waiver_command)
        )
        results.append(
            _result(
                specs["validation_waivers"],
                code,
                output,
                waiver_command,
                note="library-level verification; no signing capability acquired",
            )
        )
        if caller.run_generated_guard:
            guard_command = [
                "guard-generated",
                "--repo",
                str(repo),
                "--base-ref",
                args.base_ref,
                "--head-ref",
                "HEAD",
                "--corpus-path",
                str(paths["corpus"]),
                "--expected-encoder-checkout",
                str(paths["encode"]),
            ]
            results.append(
                _result(
                    specs["guard_generated"],
                    *_run_cli(guard_command),
                    guard_command,
                    note="library-level verification; --apply is never available to ci",
                )
            )
        else:
            results.append(
                _result(
                    specs["guard_generated"],
                    0,
                    "Disabled by caller run-generated-guard.\n",
                    [],
                )
            )
    selection_output = f"RuleSpec validation mode: {selection.mode}\n" + "\n".join(
        f"- {path.relative_to(repo)}" for path in selection.rulespec_files
    )
    results.append(
        _result(
            specs["select_targets"],
            0,
            selection_output,
            ["selection", "--base-ref", args.base_ref, "--roots", *roots],
        )
    )
    skipped = set()
    gaps_path = repo / "known-validation-gaps.yaml"
    if gaps_path.is_file():
        gaps = yaml.safe_load(gaps_path.read_text()) or {}
        entries = gaps.get("validate_failures", {}) if isinstance(gaps, dict) else {}
        skipped = set(entries if isinstance(entries, (dict, list)) else ())
    validate_failures, validate_output = [], []
    for path in selection.rulespec_files:
        if path.relative_to(repo).as_posix() in skipped:
            continue
        command = [
            "validate",
            str(path),
            "--skip-reviewers",
            "--corpus-path",
            str(paths["corpus"]),
            "--axiom-rules-engine-path",
            str(paths["engine"]),
        ]
        code, output = _run_cli(
            command,
            environment={
                "AXIOM_RULESPEC_REPO_ROOTS": os.pathsep.join(
                    (str(repo), str(paths["rulespec_us"]))
                )
            },
        )
        validate_output.append(output)
        if code:
            validate_failures.append(path.relative_to(repo).as_posix())
    results.append(
        GateResult(
            "validate",
            specs["validate"].name,
            "FAIL" if validate_failures else "PASS",
            ["validate", "{selected files}"],
            validate_failures,
            "".join(validate_output),
            "direct subcommand under library-level release verification",
        )
    )
    test_failures, test_output = [], []
    grouped: dict[str, list[Path]] = {}
    for path in selection.test_files:
        relative = path.relative_to(repo)
        module = (
            relative.as_posix()
            .replace(".test.yaml", ".yaml")
            .replace(".test.yml", ".yml")
        )
        if module not in skipped:
            grouped.setdefault(relative.parts[0], []).append(relative)
    for jurisdiction, files in grouped.items():
        command = [
            "test",
            "--root",
            str(repo / jurisdiction),
            "--axiom-rules-engine-path",
            str(paths["engine"]),
            *(str(Path(*file.parts[1:])) for file in files),
        ]
        code, output = _run_cli(
            command,
            environment={
                "AXIOM_RULESPEC_REPO_ROOTS": os.pathsep.join(
                    (str(repo), str(paths["rulespec_us"]))
                )
            },
        )
        test_output.append(output)
        if code:
            test_failures.extend(file.as_posix() for file in files)
    results.append(
        GateResult(
            "companion_tests",
            specs["companion_tests"].name,
            "FAIL" if test_failures else "PASS",
            ["test", "{selected tests}"],
            test_failures,
            "".join(test_output),
        )
    )
    proof_failures, proof_output = [], []
    for path in selection.rulespec_files:
        command = ["proof-validate", str(path), "--corpus-path", str(paths["corpus"])]
        code, output = _run_cli(command)
        proof_output.append(output)
        if code:
            proof_failures.append(path.relative_to(repo).as_posix())
    results.append(
        GateResult(
            "proof_validate",
            specs["proof_validate"].name,
            "FAIL" if proof_failures else "PASS",
            ["proof-validate", "{selected files}"],
            proof_failures,
            "".join(proof_output),
            "direct subcommand under library-level release verification",
        )
    )
    all_rules = tuple(
        sorted(
            path
            for root in roots
            if (repo / root).is_dir()
            for path in (repo / root).rglob("*.y*ml")
            if ".test." not in path.name
            and (
                not gate_parameters.exclude_programs_from_money_atom_check
                or "/programs/" not in path.as_posix()
            )
        )
    )
    if caller.run_money_atom_check and all_rules:
        command = [
            "proof-validate",
            *map(str, all_rules),
            "--money-atoms-only",
            "--corpus-path",
            str(paths["corpus"]),
        ]
        if (repo / "known-missing-money-atoms.yaml").is_file():
            command += ["--ratchet-file", str(repo / "known-missing-money-atoms.yaml")]
        results.append(
            _result(
                specs["money_atoms"],
                *_run_cli(command),
                command,
                note="direct subcommand under library-level release verification",
            )
        )
    else:
        results.append(
            _result(
                specs["money_atoms"],
                0,
                "No RuleSpec YAML files found or gate disabled.\n",
                [],
            )
        )
    if selection.mode == "full-toolchain-bump":
        command = [
            "oracle-coverage",
            "--root",
            str(repo),
            "--fail-on-unmapped",
            "--fail-on-untested-comparable",
            "--limit",
            "50",
        ]
        results.append(_result(specs["oracle_coverage"], *_run_cli(command), command))
        results.append(
            _result(
                specs["changed_oracle_coverage"],
                0,
                "Not used for full-toolchain-bump mode.\n",
                [],
            )
        )
    else:
        results.append(
            _result(
                specs["oracle_coverage"],
                0,
                "Full oracle coverage is not selected for changed-file mode.\n",
                [],
            )
        )
        results.append(
            _result(
                specs["changed_oracle_coverage"],
                *_classifier_gate(
                    repo,
                    selection,
                    paths["encode"],
                    caller.refs["encode"],
                ),
                # This is deliberately the already-authenticated local checkout,
                # archived at the caller pin; no fresh checkout or ambient runtime
                # classifier is used.
                ["oracle-coverage", "--root", str(repo), "--json"],
                note=(
                    "classifier executed from encode pin "
                    f"{caller.refs['encode']} in {paths['encode']}"
                ),
            )
        )
    return results


# ---------------------------------------------------------------------------
# Workflow pins from 0effa6a5 on: execute the pinned workflow's own scripts.
# ---------------------------------------------------------------------------

RULESPEC_CONTENT_ROOTS = ("statutes", "regulations", "policies", "legislation")
_HEREDOC_START_RE = re.compile(r"<<'(?P<tag>[A-Za-z_][A-Za-z0-9_]*)'")
_GITHUB_REMOTE_RE = re.compile(
    r"github\.com[:/](?P<slug>[A-Za-z0-9._-]+/[A-Za-z0-9._-]+?)(?:\.git)?/?$"
)
# Workers start isolated (-I: no cwd, PYTHONPATH or user site on sys.path),
# import the encoder from the parent's exact package root, and refuse to run
# if the module they loaded is not the one the parent verified.
_ISOLATED_CLI_BOOTSTRAP = (
    "import json, sys\n"
    "request = json.load(sys.stdin)\n"
    "sys.path.insert(0, request['package_root'])\n"
    "import axiom_encode.ci_parity as ci_parity\n"
    "if ci_parity.__file__ != request['module_file']:\n"
    "    sys.exit('isolated ci worker imported ' + str(ci_parity.__file__)"
    " + ', not ' + request['module_file'])\n"
    "sys.exit(ci_parity._isolated_cli_main(request))\n"
)


def _git_blob_id(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def _heredoc_bodies(run: str) -> list[str]:
    """Return every quoted heredoc body in a bash ``run`` script, in order."""

    lines = run.split("\n")
    bodies: list[str] = []
    index = 0
    while index < len(lines):
        match = _HEREDOC_START_RE.search(lines[index])
        index += 1
        if match is None:
            continue
        tag = match.group("tag")
        try:
            end = lines.index(tag, index)
        except ValueError as exc:
            raise ValueError(f"Unterminated heredoc {tag} in workflow step") from exc
        bodies.append("".join(f"{line}\n" for line in lines[index:end]))
        index = end + 1
    return bodies


def _single_quoted_python(run: str) -> str:
    """Return the program of the one ``python3 -c '...'`` in a ``run`` script."""

    marker = "python3 -c '"
    start = run.index(marker) + len(marker)
    end = run.index("\n'", start)
    return run[start : end + 1]


class PinnedWorkflow:
    """The packaged reusable workflow for one supported pin, blob-verified."""

    def __init__(self, workflow_sha: str) -> None:
        pin = SUPPORTED_WORKFLOW_PINS[workflow_sha]
        path = WORKFLOW_DIRECTORY / pin.fixture
        data = path.read_bytes()
        actual = _git_blob_id(data)
        if actual != pin.workflow_blob:
            raise ValueError(
                f"Packaged workflow {path} is blob {actual}, not the "
                f"validate-rulespec@{workflow_sha} blob {pin.workflow_blob}"
            )
        self.sha = workflow_sha
        self.pin = pin
        self.path = path
        self.payload = yaml.safe_load(data)

    def step(self, job: str, name: str) -> dict[str, Any]:
        steps = self.payload["jobs"][job]["steps"]
        matches = [
            step
            for step in steps
            if isinstance(step, dict) and step.get("name") == name
        ]
        if len(matches) != 1:
            raise ValueError(
                f"validate-rulespec@{self.sha[:8]} job {job} has "
                f"{len(matches)} steps named {name!r}"
            )
        return matches[0]

    def python(self, job: str, name: str, occurrence: int = 0) -> str:
        return _heredoc_bodies(self.step(job, name)["run"])[occurrence]

    def inline_python(self, job: str, name: str) -> str:
        return _single_quoted_python(self.step(job, name)["run"])


def _run_embedded_python(
    script: str,
    *,
    cwd: Path,
    argv: Sequence[str] = (),
    environment: Mapping[str, str] | None = None,
    stdin: bytes = b"",
) -> tuple[int, str, str]:
    """Run one of the workflow's inline Python programs as the step would.

    The ambient GitHub Actions context is withheld; the caller passes exactly
    the step environment the simulated event defines.
    """

    child_environment = {
        name: value
        for name, value in _git_environment().items()
        if not name.startswith(("GITHUB_", "RUNNER_", "PYTHON"))
    }
    child_environment.update(environment or {})
    result = subprocess.run(
        [sys.executable, "-c", script, *argv],
        cwd=cwd,
        env=child_environment,
        input=stdin,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    return (
        result.returncode,
        result.stdout.decode("utf-8", "replace"),
        result.stderr.decode("utf-8", "replace"),
    )


@dataclass(frozen=True, slots=True)
class PullRequestSimulation:
    """The pull_request event ci reproduces: HEAD proposed against a base."""

    base_sha: str
    head_sha: str
    base_branch: str
    repository: str
    number: int | None
    base_is_ancestor: bool

    @property
    def ref(self) -> str:
        return f"refs/pull/{self.number}/merge" if self.number is not None else ""

    def github_environment(self, repo: Path) -> dict[str, str]:
        return {
            "GITHUB_EVENT_NAME": PULL_REQUEST_EVENT,
            "GITHUB_REF": self.ref,
            "GITHUB_REPOSITORY": self.repository,
            "GITHUB_SHA": self.head_sha,
            "GITHUB_WORKSPACE": str(repo),
        }


def _github_repository(repo: Path) -> str:
    """Return owner/name of the checkout's origin, or "" when not on GitHub."""

    result = _git(repo, "remote", "get-url", "origin", check=False)
    match = _GITHUB_REMOTE_RE.search(result.stdout.strip())
    return match.group("slug") if result.returncode == 0 and match else ""


def simulate_pull_request(
    repo: Path, base_ref: str, number: int | None = None
) -> PullRequestSimulation:
    base = _git(repo, "rev-parse", "--verify", f"{base_ref}^{{commit}}", check=False)
    if base.returncode or SHA_RE.fullmatch(base.stdout.strip()) is None:
        raise ValueError(f"--base-ref {base_ref} does not name a commit in {repo}")
    head = _git(repo, "rev-parse", "--verify", "HEAD^{commit}")
    base_sha, head_sha = base.stdout.strip(), head.stdout.strip()
    # The pull request's base branch: a symbolic ref such as @{upstream}
    # names its branch, while a bare SHA keeps the given text.
    symbolic = _git(repo, "rev-parse", "--symbolic-full-name", base_ref, check=False)
    branch = symbolic.stdout.strip() if symbolic.returncode == 0 else ""
    branch = branch or base_ref
    if branch.startswith("refs/remotes/"):
        branch = branch.removeprefix("refs/remotes/").split("/", 1)[-1]
    else:
        for prefix in ("refs/heads/", "origin/"):
            if branch.startswith(prefix):
                branch = branch[len(prefix) :]
                break

    ancestor = _git(
        repo, "merge-base", "--is-ancestor", base_sha, head_sha, check=False
    )
    return PullRequestSimulation(
        base_sha=base_sha,
        head_sha=head_sha,
        base_branch=branch,
        repository=_github_repository(repo),
        number=number,
        base_is_ancestor=ancestor.returncode == 0,
    )


def verify_dependency_inputs(caller: CallerConfig) -> None:
    """Mirror the workflow's "Validate immutable dependency inputs" step."""

    for dependency in DEPENDENCY_INPUTS:
        value = caller.refs.get(dependency, "")
        if not isinstance(value, str) or SHA_RE.fullmatch(value) is None:
            raise ValueError(
                "Dependency inputs must be exact 40-character lowercase commit "
                f"SHAs: {value}"
            )
    python_version = str(caller.inputs.get("python-version", "3.14"))
    if re.fullmatch(r"3\.[0-9]+", python_version) is None:
        raise ValueError(
            f"python-version must be a bare major.minor like 3.14: {python_version}"
        )
    registry_url = str(caller.inputs.get("corpus-release-registry-url", ""))
    anon_key = str(caller.inputs.get("corpus-release-registry-anon-key", ""))
    base_url = str(
        caller.inputs.get("corpus-release-base-url", caller.release_base_url)
    )
    if registry_url:
        if not registry_url.startswith("https://"):
            raise ValueError("corpus-release-registry-url must use HTTPS")
        if not anon_key:
            raise ValueError(
                "corpus-release-registry-anon-key is required with "
                "corpus-release-registry-url"
            )
    else:
        if anon_key:
            raise ValueError(
                "corpus-release-registry-anon-key requires corpus-release-registry-url"
            )
        if not base_url.startswith("https://"):
            raise ValueError("corpus-release-base-url must use HTTPS")


def resolve_workflow_toolchain(
    workflow: PinnedWorkflow, repo: Path, simulation: PullRequestSimulation
) -> None:
    """Run the workflow's strict "Resolve RuleSpec toolchain" script."""

    with tempfile.TemporaryDirectory(prefix="axiom-ci-toolchain-") as temp_name:
        code, stdout, stderr = _run_embedded_python(
            workflow.python("validate", "Resolve RuleSpec toolchain"),
            cwd=repo,
            environment={
                **simulation.github_environment(repo),
                "GITHUB_OUTPUT": str(Path(temp_name) / "output"),
                "PR_BASE_SHA": simulation.base_sha,
            },
        )
    if code:
        raise ValueError((stderr or stdout).strip() or "RuleSpec toolchain error")


@dataclass(frozen=True, slots=True)
class ShardPlan:
    """The shards job's outputs for the simulated pull request."""

    matrix: tuple[str, ...]
    first: str
    roots: str
    allowed_extra: str
    matrix_json: str
    scope: str

    def shard_roots(self, shard: str, validate_roots_input: str) -> str:
        """Mirror the "Resolve shard validation roots" step."""

        return validate_roots_input if shard == "__all__" else shard


def compute_shard_plan(
    workflow: PinnedWorkflow,
    repo: Path,
    simulation: PullRequestSimulation,
    validate_roots_input: str,
    guard_programs_root: bool,
) -> ShardPlan:
    """Mirror the shards job's "Compute validation shards" step."""

    if any(character in validate_roots_input for character in "*?["):
        raise ValueError(
            f"validate-roots {validate_roots_input!r} contains glob characters, "
            "which the workflow's unquoted expansion would reinterpret"
        )
    scope = "explicit validate-roots"
    if validate_roots_input == "auto":
        all_shards = [
            child.name
            for child in _glob_directories(repo)
            if not child.name.startswith("_")
            and re.fullmatch(r"[a-z]{2}(-[a-z0-9-]+)*", child.name)
            and any((child / marker).is_dir() for marker in RULESPEC_CONTENT_ROOTS)
        ]
        if not all_shards:
            raise ValueError("validate-roots=auto found no jurisdiction directories")
        roots = " ".join(all_shards)
        if guard_programs_root:
            roots = f"{roots} programs"
            allowed_extra = "sources"
        else:
            allowed_extra = "sources programs"
        code, stdout, stderr = _run_embedded_python(
            workflow.python("shards", "Compute validation shards"),
            cwd=repo,
            environment={
                "ALL_SHARDS": " ".join(all_shards),
                "EVENT_NAME": PULL_REQUEST_EVENT,
                "PR_BASE_SHA": simulation.base_sha,
                "HEAD_SHA": simulation.head_sha,
            },
        )
        # mapfile over a process substitution ignores the script's exit status.
        shards = stdout.splitlines()
        scope = (stderr.strip().splitlines() or [f"scope exit {code}"])[-1]
        if not shards:
            scope = f"{scope}; empty selection fell back to the full matrix"
            shards = list(all_shards)
    else:
        shards = ["__all__"]
        roots = validate_roots_input
        if guard_programs_root and " programs " not in f" {roots} ":
            roots = f"{roots} programs"
        allowed_extra = "sources"
    matrix = tuple(line.strip() for line in shards if line.strip())
    return ShardPlan(
        matrix=matrix,
        first=shards[0],
        roots=roots,
        allowed_extra=allowed_extra,
        matrix_json=json.dumps(list(matrix)),
        scope=scope,
    )


class _HttpsOnlyRedirects(urllib.request.HTTPRedirectHandler):
    """Follow redirects only to HTTPS, like curl --proto '=https'."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if not newurl.startswith("https://"):
            raise urllib.error.HTTPError(
                newurl, code, "refusing a non-HTTPS redirect", headers, fp
            )
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _fetch_https(url: str, headers: Mapping[str, str]) -> bytes:
    """GET an HTTPS URL with a bounded read and an explicit User-Agent.

    The r2.dev mirror's Cloudflare front returns 403 to urllib's default
    "Python-urllib/3.x" User-Agent, so identify the client explicitly.
    """

    if not url.startswith("https://"):
        raise ValueError(f"Refusing to fetch a non-HTTPS URL: {url}")
    request = urllib.request.Request(
        url, headers={"User-Agent": f"axiom-encode/{__version__}", **headers}
    )
    opener = urllib.request.build_opener(_HttpsOnlyRedirects)
    with opener.open(request, timeout=60) as response:
        data = response.read(MAX_RELEASE_DOWNLOAD_BYTES + 1)
    if len(data) > MAX_RELEASE_DOWNLOAD_BYTES:
        raise ValueError(
            f"Release object download exceeds {MAX_RELEASE_DOWNLOAD_BYTES} bytes"
        )
    return data


def acquire_workflow_release_object(
    workflow: PinnedWorkflow,
    toolchain: RuleSpecToolchain,
    corpus_path: Path,
    caller: CallerConfig,
    *,
    offline: bool,
    fetcher: Callable[[str, Mapping[str, str]], bytes] | None = None,
) -> tuple[Path, str]:
    """Mirror "Fetch pinned signed corpus release object" for this pin.

    As in CI, the object is downloaded on every run: from the registry alone
    when the caller sets a registry URL, otherwise from the object mirror. The
    workflow's own verifier then checks it and rewrites it at the cache path.
    Only ``offline`` uses the cached copy, re-verified the same way.
    """

    release = toolchain.corpus_release
    content_sha256 = toolchain.corpus_release_content_sha256
    destination = corpus_path / "releases" / release / f"{content_sha256}.json"
    registry_url = str(caller.inputs.get("corpus-release-registry-url", ""))
    base_url = str(
        caller.inputs.get("corpus-release-base-url", caller.release_base_url)
    )
    fetch = fetcher or _fetch_https
    verifier = workflow.python("validate", "Fetch pinned signed corpus release object")

    def verify(raw: bytes, acquisition: str) -> tuple[int, str]:
        # The verifier rewrites "<object>.json.tmp" and renames it to
        # "<object>.json" beside it. CI gives every job its own workspace;
        # locally several runs can share one corpus checkout, so each stages
        # the object in a private directory, and the verified file replaces
        # the cached copy in a single rename.
        destination.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=".ci-release-", dir=destination.parent))
        try:
            temporary = staging / f"{destination.name}.tmp"
            temporary.write_bytes(raw)
            code, stdout, stderr = _run_embedded_python(
                verifier,
                cwd=corpus_path,
                argv=(str(temporary), release, content_sha256, acquisition),
            )
            if not code:
                (staging / destination.name).replace(destination)
            return code, (stderr or stdout).strip()
        finally:
            shutil.rmtree(staging, ignore_errors=True)

    if offline:
        # CI always downloads the object. Offline, the cached copy stands in
        # for that download and is re-verified exactly as a download would be.
        if not destination.is_file():
            raise ValueError(
                f"--offline requires pinned corpus release object: {destination}"
            )
        code, message = verify(destination.read_bytes(), "object")
        if code:
            raise ValueError(
                f"Cached corpus release object failed verification: {message}"
            )
        return destination, (
            f"corpus release {release}@{content_sha256[:12]} from cached "
            f"{destination} (--offline; CI downloads it)"
        )
    if registry_url:
        anon_key = str(caller.inputs["corpus-release-registry-anon-key"])
        registry = registry_url[:-1] if registry_url.endswith("/") else registry_url
        url = (
            f"{registry}/rest/v1/release_objects?select=release_object"
            f"&release_name=eq.{release}&content_sha256=eq.{content_sha256}&limit=2"
        )
        raw = _acquire(
            fetch,
            url,
            {
                "apikey": anon_key,
                "Authorization": f"Bearer {anon_key}",
                "Accept-Profile": "corpus",
            },
        )
        acquisition = "registry"
        source = f"registry {registry}"
    else:
        base = base_url[:-1] if base_url.endswith("/") else base_url
        url = f"{base}/releases/{release}/{content_sha256}.json"
        raw = _acquire(fetch, url, {})
        acquisition = "object"
        source = f"mirror {url}"
    code, message = verify(raw, acquisition)
    if code:
        raise ValueError(message)
    return destination, f"corpus release {release}@{content_sha256[:12]} from {source}"


def _acquire(
    fetch: Callable[[str, Mapping[str, str]], bytes],
    url: str,
    headers: Mapping[str, str],
) -> bytes:
    try:
        return fetch(url, headers)
    except (OSError, ValueError) as exc:
        raise ValueError(f"Corpus release acquisition error: {url}: {exc}") from exc


@dataclass(frozen=True, slots=True)
class CliInvocation:
    arguments: tuple[str, ...]
    cwd: Path
    supervised: bool = False
    environment: Mapping[str, str] | None = None


def _run_cli_batch(
    invocations: Sequence[CliInvocation],
    *,
    jobs: int,
    keyring: Sequence[str],
    signing_roots: Mapping[str, str] | None = None,
) -> list[tuple[int, str]]:
    """Run independent encoder invocations, in-process or in isolated workers.

    Each CI invocation is its own process; with ``jobs`` above one, ci starts
    that many isolated interpreters at a time. The workers receive the
    verification keyring on stdin, never through the environment, and cap the
    waiver audit's own fan-out so the machine is not oversubscribed; the
    audit's fingerprints are independent of that worker count by design.
    """

    if jobs <= 1 or len(invocations) <= 1:
        return [
            _run_cli(
                invocation.arguments,
                environment=dict(invocation.environment or {}),
                cwd=invocation.cwd,
                supervised=invocation.supervised,
            )
            for invocation in invocations
        ]
    worker_cap = max(1, (os.cpu_count() or 1) // jobs)
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        return list(
            pool.map(
                lambda invocation: _run_cli_isolated(
                    invocation, keyring, worker_cap, signing_roots or {}
                ),
                invocations,
            )
        )


def _run_cli_isolated(
    invocation: CliInvocation,
    keyring: Sequence[str],
    worker_cap: int,
    signing_roots: Mapping[str, str] | None = None,
) -> tuple[int, str]:
    module_file = str(Path(__file__).resolve())
    request = {
        "package_root": str(Path(module_file).parents[1]),
        "module_file": module_file,
        "keyring": list(keyring),
        "signing_roots": dict(signing_roots or {}),
        "arguments": list(invocation.arguments),
        "cwd": str(invocation.cwd),
        "supervised": invocation.supervised,
        "environment": {
            **(invocation.environment or {}),
            WAIVER_AUDIT_WORKERS_ENV: str(worker_cap),
        },
    }
    result = subprocess.run(
        [sys.executable, "-I", "-c", _ISOLATED_CLI_BOOTSTRAP],
        input=json.dumps(request).encode("utf-8"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=invocation.cwd,
        env=_git_environment(),
        check=False,
    )
    return result.returncode, (
        result.stdout.decode("utf-8", "replace")
        + result.stderr.decode("utf-8", "replace")
    )


def _isolated_cli_main(request: Mapping[str, Any]) -> int:
    """Worker entrypoint for _run_cli_isolated; holds no signing capability."""

    keyring = list(request["keyring"])
    signing_roots = dict(request.get("signing_roots") or {})
    with local_corpus_release_verification(
        keyring[0],
        retired_public_keys=keyring[1:],
        apply_public_key=signing_roots.get("apply"),
        eval_public_key=signing_roots.get("eval"),
    ):
        code, output = _run_cli(
            request["arguments"],
            environment=dict(request["environment"]),
            cwd=Path(request["cwd"]),
            supervised=bool(request["supervised"]),
        )
    sys.stdout.write(output)
    sys.stdout.flush()
    return code


def _find_regular_files(
    repo: Path, start: str, *, prune: frozenset[str] = frozenset()
) -> Iterator[str]:
    """Yield ``find <start> -type f`` paths as find prints them (no symlinks).

    Paths in ``prune`` are skipped with everything below them, like
    ``find ... \\( -path <p> ... \\) -prune``.
    """

    root = repo / start if start != "." else repo
    try:
        if stat.S_ISLNK(root.lstat().st_mode) or not root.is_dir():
            return
    except OSError:
        return
    pending = [(root, start)]
    while pending:
        directory, printed = pending.pop()
        try:
            entries = sorted(os.scandir(directory), key=lambda entry: entry.name)
        except OSError:
            continue
        for entry in entries:
            child = f"{printed}/{entry.name}"
            if child in prune:
                continue
            if entry.is_dir(follow_symlinks=False):
                pending.append((Path(entry.path), child))
            elif entry.is_file(follow_symlinks=False):
                yield child


def _find_named(
    repo: Path, patterns: Sequence[str], prune: frozenset[str]
) -> list[str]:
    """``find . \\( prune \\) -prune -o \\( -name p ... \\) -print`` (any type)."""

    matches: list[str] = []
    pending = [(repo, ".")]
    while pending:
        directory, printed = pending.pop()
        try:
            entries = sorted(os.scandir(directory), key=lambda entry: entry.name)
        except OSError:
            continue
        for entry in entries:
            child = f"{printed}/{entry.name}"
            if child in prune:
                continue
            if any(fnmatch.fnmatchcase(entry.name, pattern) for pattern in patterns):
                matches.append(child)
            if entry.is_dir(follow_symlinks=False):
                pending.append((Path(entry.path), child))
    return matches


_FIND_PRUNED = frozenset({"./.git", "./_axiom", "./.venv", "./.pytest_cache"})


def _changed_path_list(repo: Path, base: str, head: str, diff_filter: str) -> list[str]:
    result = subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "diff",
            "--name-only",
            "-z",
            "--no-renames",
            f"--diff-filter={diff_filter}",
            base,
            head,
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=_git_environment(),
    )
    return [os.fsdecode(path) for path in result.stdout.split(b"\0") if path]


@dataclass(frozen=True, slots=True)
class ShardSelection:
    shard: str
    roots: tuple[str, ...]
    rulespec_files: tuple[str, ...]
    test_files: tuple[str, ...]


@dataclass(slots=True)
class WorkflowRun:
    """Everything one embedded-script parity run shares across its gates."""

    caller: CallerConfig
    workflow: PinnedWorkflow
    paths: dict[str, Path]
    repo: Path
    simulation: PullRequestSimulation
    plan: ShardPlan
    validate_roots_input: str
    keyring: tuple[str, ...]
    jobs: int = 1
    # Verification-only "apply"/"eval" roots (base64), as the supervisor has.
    signing_roots: Mapping[str, str] = field(default_factory=dict)
    temp: Path = field(default_factory=Path)
    migration_authorized: str = ""
    migration_candidate: str = ""
    retired_skip: frozenset[str] = frozenset()
    mode: str = "changed"
    changed: tuple[str, ...] = ()
    selections: dict[str, ShardSelection] = field(default_factory=dict)
    # The skip list the validate step wrote for each shard it got that far in;
    # the companion-test step reads that file and never builds its own.
    validate_skip: dict[str, frozenset[str]] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def execute_workflow_gates(run: WorkflowRun) -> list[GateResult]:
    """Run a 0effa6a5-or-later pin's gates in the workflow's order.

    The matrix legs run one after another. A gate that runs on every leg in CI
    is reported once, with any failure labelled by its shard; first-shard-only
    gates run once over the repository-wide roots, as CI's first leg does.
    """

    specs = {spec.key: spec for spec in gate_registry_for_pin(run.workflow.sha)}
    handlers: dict[str, Callable[[WorkflowRun, GateSpec], GateResult]] = {
        "unsupported_paths": _gate_unsupported_paths,
        "unmanifested_rulespec": _gate_unmanifested_rulespec,
        "migration_authorization": _gate_migration_authorization,
        "retired_schema_freeze": _gate_retired_schema_freeze,
        "obsolete_files": _gate_obsolete_files,
        "repository_layout": _gate_repository_layout,
        "validation_waivers": _gate_validation_waivers,
        "guard_generated": _gate_guard_generated,
        "select_targets": _gate_select_targets,
        "validate": _gate_validate,
        "companion_tests": _gate_companion_tests,
        "proof_validate": _gate_proof_validate,
        "money_atoms": _gate_money_atoms,
        "oracle_coverage": _gate_oracle_coverage,
        "changed_oracle_coverage": _gate_changed_oracle_coverage,
        "repository_tests": _gate_repository_tests,
    }
    results: list[GateResult] = []
    with tempfile.TemporaryDirectory(prefix="axiom-ci-") as temp_name:
        run.temp = Path(temp_name)
        for key in run.workflow.pin.gates:
            results.append(handlers[key](run, specs[key]))
    return results


def _step_result(
    spec: GateSpec,
    code: int,
    stdout: str,
    stderr: str,
    command: Sequence[str],
    *,
    note: str | None = None,
) -> GateResult:
    return _result(spec, code, stdout + stderr, command, note=note)


def _aggregate(
    spec: GateSpec,
    run: WorkflowRun,
    outcomes: Sequence[tuple[str, int, str]],
    command: Sequence[str],
    *,
    note: str | None = None,
) -> GateResult:
    """Combine per-shard outcomes; failures are labelled by shard."""

    labelled = len(run.plan.matrix) > 1
    failures: list[str] = []
    output: list[str] = []
    for shard, code, text in outcomes:
        prefix = f"[{shard}] " if labelled else ""
        output.append("".join(f"{prefix}{line}\n" for line in text.splitlines()))
        if code:
            lines = [f"{prefix}{line}" for line in text.splitlines() if line.strip()]
            failures.extend(lines or [f"{prefix}exit status {code}"])
    return GateResult(
        spec.key,
        spec.name,
        "FAIL" if any(code for _, code, _ in outcomes) else "PASS",
        list(command),
        failures,
        "".join(output),
        note,
    )


def _render_boolean(value: Any) -> str:
    """Render a boolean input the way ${{ inputs.<name> }} does."""

    return "true" if value is True else "false" if value is False else str(value)


def _render_number(value: Any) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _gate_unsupported_paths(run: WorkflowRun, spec: GateSpec) -> GateResult:
    command = ["tracked-paths", run.simulation.base_sha, run.simulation.head_sha]
    listed = []
    for arguments in (
        ["ls-files", "-z"],
        [
            "diff",
            "--name-only",
            "-z",
            "--no-renames",
            "--diff-filter=ACMRTD",
            run.simulation.base_sha,
            run.simulation.head_sha,
        ],
    ):
        result = subprocess.run(
            ["git", "-C", str(run.repo), *arguments],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            env=_git_environment(),
        )
        if result.returncode:
            return _result(spec, 1, result.stderr.decode("utf-8", "replace"), command)
        listed.append(result.stdout)
    code, stdout, stderr = _run_embedded_python(
        run.workflow.inline_python("shards", spec.name),
        cwd=run.repo,
        environment={
            **run.simulation.github_environment(run.repo),
            "BASE_SHA": run.simulation.base_sha,
        },
        stdin=b"".join(listed),
    )
    if not code and not stdout:
        stdout = "No unsupported tracked or changed paths.\n"
    return _step_result(spec, code, stdout, stderr, command)


def _gate_unmanifested_rulespec(run: WorkflowRun, spec: GateSpec) -> GateResult:
    code, stdout, stderr = _run_embedded_python(
        run.workflow.python("shards", spec.name),
        cwd=run.repo,
        environment={
            **run.simulation.github_environment(run.repo),
            "BASE_SHA": run.simulation.base_sha,
            "EVENT_NAME": PULL_REQUEST_EVENT,
            "RUN_GENERATED_GUARD": _render_boolean(
                run.caller.inputs["run-generated-guard"]
            ),
            "SCAN_MODE": "diff",
        },
    )
    return _step_result(
        spec,
        code,
        stdout,
        stderr,
        ["manifest-precheck", f"{run.simulation.base_sha}...{run.simulation.head_sha}"],
    )


def _gate_migration_authorization(run: WorkflowRun, spec: GateSpec) -> GateResult:
    inputs = run.caller.inputs
    code, stdout, stderr = _run_embedded_python(
        run.workflow.python("validate", spec.name),
        cwd=run.repo,
        environment={
            **run.simulation.github_environment(run.repo),
            "AUTHORIZATION_PATH": str(inputs["migration-authorization-path"]),
            "BASE_SHA": run.simulation.base_sha,
            "EVENT_NAME": PULL_REQUEST_EVENT,
            "PR_BASE_REF": run.simulation.base_branch,
            "PR_HEAD_SHA": run.simulation.head_sha,
            "PR_NUMBER": ""
            if run.simulation.number is None
            else str(run.simulation.number),
            "RETIRED_SCHEMA_BOOTSTRAP": str(inputs["retired-schema-bootstrap-sha256"]),
            "VALIDATION_WAIVER_BOOTSTRAP": str(
                inputs["validation-waiver-bootstrap-sha256"]
            ),
            "RUN_GENERATED_GUARD": _render_boolean(inputs["run-generated-guard"]),
        },
    )
    outputs = dict(line.split("=", 1) for line in stdout.splitlines() if "=" in line)
    if not code:
        run.migration_authorized = outputs.get("authorized", "")
        run.migration_candidate = outputs.get("candidate", "")
    note = None
    if code and run.simulation.number is None:
        note = "pass --pull-request N to reproduce a pull request's authorization"
    return _step_result(
        spec,
        code,
        stdout,
        stderr,
        ["migration-authorization", str(inputs["migration-authorization-path"])],
        note=note,
    )


def _gate_retired_schema_freeze(run: WorkflowRun, spec: GateSpec) -> GateResult:
    inputs = run.caller.inputs
    skip_path = run.temp / "retired-schema-skip.txt"
    # Every leg checks the head freeze; the first leg adds the base, inventory
    # and bootstrap checks. Running the first leg's superset once decides both.
    code, stdout, stderr = _run_embedded_python(
        run.workflow.python("validate", spec.name),
        cwd=run.repo,
        argv=(str(skip_path),),
        environment={
            **run.simulation.github_environment(run.repo),
            "BASE_SHA": run.simulation.base_sha,
            "BOOTSTRAP_SHA256": str(inputs["retired-schema-bootstrap-sha256"]),
            "CALLER_REPOSITORY": run.simulation.repository,
            "EVENT_NAME": PULL_REQUEST_EVENT,
            "HEAD_MESSAGE": "",
            "PR_NUMBER": ""
            if run.simulation.number is None
            else str(run.simulation.number),
            "REF_NAME": run.simulation.ref,
            "IS_FIRST_SHARD": "true",
            "RUN_GENERATED_GUARD": _render_boolean(inputs["run-generated-guard"]),
            "ALLOW_PREFREEZE": _render_boolean(
                inputs["allow-retired-schema-prefreeze"]
            ),
            "MIGRATION_AUTHORIZED": run.migration_authorized,
        },
    )
    if not code and skip_path.is_file():
        run.retired_skip = frozenset(
            line for line in skip_path.read_text(encoding="utf-8").splitlines() if line
        )
    return _step_result(
        spec,
        code,
        stdout,
        stderr,
        ["retired-schema-freeze", ".axiom/retired-schema-freeze.json"],
    )


def _gate_obsolete_files(run: WorkflowRun, spec: GateSpec) -> GateResult:
    obsolete = "rac"
    matches = _find_named(
        run.repo, (f"*.{obsolete}", f"*.{obsolete}.test"), _FIND_PRUNED
    )
    return _result(
        spec,
        1 if matches else 0,
        "\n".join(matches) + "\n" if matches else "No obsolete generated files.\n",
        ["find", ".", f"*.{obsolete}", f"*.{obsolete}.test"],
        note=f"first shard ({run.plan.first})",
    )


def _gate_repository_layout(run: WorkflowRun, spec: GateSpec) -> GateResult:
    command = ["layout", run.plan.roots, run.plan.allowed_extra]
    if (run.repo / ".axiom" / "repository-structure.yaml").is_file():
        code, stdout, stderr = _run_embedded_python(
            run.workflow.python("validate", spec.name), cwd=run.repo
        )
        return _step_result(spec, code, stdout, stderr, command)
    disallowed = [
        f"{root}/"
        for root in ("statute", "regulation", "policy")
        if (run.repo / root).exists()
    ]
    disallowed += [
        path
        for path in _find_all_regular(run.repo, _FIND_PRUNED)
        if PurePosixPath(path).name in {"parameters.yaml", "tests.yaml"}
    ]
    disallowed += [
        path
        for path in _find_regular_files(run.repo, "tests")
        if fnmatch.fnmatchcase(PurePosixPath(path).name, "*.yaml")
    ]
    allowed = frozenset(
        f"./{root}"
        for root in (*run.plan.roots.split(), *run.plan.allowed_extra.split())
    )
    disallowed += [
        path
        for path in _find_all_regular(run.repo, _FIND_PRUNED | {"./.github"} | allowed)
        if (
            fnmatch.fnmatchcase(PurePosixPath(path).name, "*.yaml")
            or fnmatch.fnmatchcase(PurePosixPath(path).name, "*.yml")
        )
        and PurePosixPath(path).name
        not in {"known-dangling.yaml", "known-validation-gaps.yaml"}
    ]
    if disallowed:
        return _result(
            spec,
            1,
            "Repository layout is not allowed. Use RuleSpec YAML under statutes/, "
            "regulations/, or policies/.\n" + "\n".join(disallowed) + "\n",
            command,
        )
    return _result(spec, 0, "Repository layout is allowed.\n", command)


def _find_all_regular(repo: Path, prune: frozenset[str]) -> list[str]:
    return list(_find_regular_files(repo, ".", prune=prune))


def _gate_validation_waivers(run: WorkflowRun, spec: GateSpec) -> GateResult:
    """Mirror "Enforce validation waiver ratchet" on every matrix leg.

    The bootstrap guard and ratchet script see identical inputs on every leg,
    so they run once; the supervised audit runs once per partition key.
    """

    repo, base, head = run.repo, run.simulation.base_sha, run.simulation.head_sha
    bootstrap = str(run.caller.inputs["validation-waiver-bootstrap-sha256"])
    protected_base = run.temp / "protected-known-validation-gaps.yaml"
    protected_toolchain = run.temp / "protected-toolchain.toml"
    changed_paths = run.temp / "waiver-changed-paths.txt"
    audit_changed_paths = run.temp / "waiver-audit-changed-paths.txt"
    command = [
        "validation-waivers",
        "audit",
        "--partition-keys-json",
        run.plan.matrix_json,
    ]

    def fail(message: str) -> GateResult:
        return _result(spec, 1, message + "\n", command)

    waivers = repo / "known-validation-gaps.yaml"
    if bootstrap:
        if run.migration_authorized != "true":
            return fail(
                "ValidationWaiverBootstrapUnauthorized: bootstrap is not bound to "
                "an exact protected authorization"
            )
        candidate = subprocess.run(
            [
                "git",
                "-C",
                str(repo),
                "show",
                f"{run.migration_candidate}:known-validation-gaps.yaml",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            env=_git_environment(),
        )
        candidate_sha256 = hashlib.sha256(candidate.stdout).hexdigest()
        if candidate.returncode or candidate_sha256 != bootstrap:
            return fail(
                "ValidationWaiverBootstrapCandidateMismatch: candidate waiver hash "
                "does not match bootstrap"
            )
        current_sha256 = hashlib.sha256(waivers.read_bytes()).hexdigest()
        if current_sha256 != bootstrap:
            return fail(
                "ValidationWaiverBootstrapHashMismatch: known-validation-gaps.yaml "
                f"sha256 {current_sha256} != bootstrap {bootstrap}"
            )
        protected_base.write_bytes(waivers.read_bytes())
    else:
        shown = subprocess.run(
            ["git", "-C", str(repo), "show", f"{base}:known-validation-gaps.yaml"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            env=_git_environment(),
        )
        if shown.returncode:
            return fail(
                "ValidationWaiverBaseMissing: protected base must contain "
                "known-validation-gaps.yaml"
            )
        protected_base.write_bytes(shown.stdout)
    diff = subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "diff",
            "--name-only",
            "-z",
            "--no-renames",
            "--diff-filter=ACMRTD",
            base,
            head,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        env=_git_environment(),
    )
    if diff.returncode:
        return fail(diff.stderr.decode("utf-8", "replace").strip())
    changed_paths.write_bytes(diff.stdout.replace(b"\0", b"\n"))
    if bootstrap:
        protected_toolchain.write_bytes(
            (repo / ".axiom" / "toolchain.toml").read_bytes()
        )
    else:
        shown = subprocess.run(
            ["git", "-C", str(repo), "show", f"{base}:.axiom/toolchain.toml"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            env=_git_environment(),
        )
        if shown.returncode:
            return fail(
                "ValidationWaiverBaseToolchainMissing: protected base must contain "
                ".axiom/toolchain.toml"
            )
        protected_toolchain.write_bytes(shown.stdout)
    code, stdout, stderr = _run_embedded_python(
        run.workflow.python("validate", spec.name),
        cwd=repo,
        argv=(
            str(protected_base),
            "known-validation-gaps.yaml",
            str(protected_toolchain),
            ".axiom/toolchain.toml",
            str(changed_paths),
            str(audit_changed_paths),
        ),
    )
    ratchet = stdout + stderr
    if code:
        return _result(spec, code, ratchet, command)
    invocations = [
        CliInvocation(
            (
                "validation-waivers",
                "audit",
                "--root",
                str(repo),
                "--corpus-path",
                str(run.paths["corpus"]),
                "--protected-base",
                str(protected_base),
                "--changed-paths",
                str(audit_changed_paths),
                "--partition-key",
                shard,
                "--partition-keys-json",
                run.plan.matrix_json,
                "--axiom-rules-engine-path",
                str(run.paths["engine"]),
            ),
            cwd=repo,
            supervised=True,
        )
        for shard in run.plan.matrix
    ]
    outcomes = _run_cli_batch(
        invocations, jobs=run.jobs, keyring=run.keyring, signing_roots=run.signing_roots
    )
    result = _aggregate(
        spec,
        run,
        [
            (shard, code, text)
            for shard, (code, text) in zip(run.plan.matrix, outcomes, strict=True)
        ],
        command,
        note="library-level verification; no signing capability acquired",
    )
    result.output = ratchet + result.output
    return result


def _gate_guard_generated(run: WorkflowRun, spec: GateSpec) -> GateResult:
    if not run.caller.inputs["run-generated-guard"]:
        return _result(spec, 0, "Disabled by caller run-generated-guard.\n", [])
    base = run.simulation.base_sha
    if run.workflow.pin.gate_parameters.exact_guard_base_ref and not SHA_RE.fullmatch(
        base
    ):
        return _result(
            spec,
            1,
            "GeneratedGuardBaseRef: base ref must be an exact 40-hex commit, "
            f"got '{base}'\n",
            [],
        )
    command = [
        "guard-generated",
        "--repo",
        str(run.repo),
        "--base-ref",
        base,
        "--head-ref",
        run.simulation.head_sha,
        "--corpus-path",
        str(run.paths["corpus"]),
        "--expected-encoder-checkout",
        str(run.paths["encode"]),
    ]
    code, output = _run_cli(command, cwd=run.repo, supervised=True)
    return _result(
        spec,
        code,
        output,
        command,
        note="library-level verification; --apply is never available to ci",
    )


def _gate_select_targets(run: WorkflowRun, spec: GateSpec) -> GateResult:
    repo, base = run.repo, run.simulation.base_sha
    try:
        run.changed = tuple(
            _changed_path_list(repo, base, run.simulation.head_sha, "ACMRTD")
        )
    except subprocess.CalledProcessError as exc:
        return _result(
            spec, 1, exc.stderr.decode("utf-8", "replace"), ["selection", base]
        )
    output: list[str] = []
    mode = "changed"
    for path in run.changed:
        if fnmatch.fnmatchcase(path, ".github/workflows/*.yml") or fnmatch.fnmatchcase(
            path, ".github/workflows/*.yaml"
        ):
            mode = "full-toolchain-bump"
            break
        if path == ".axiom/toolchain.toml":
            code, stdout, stderr = _run_embedded_python(
                run.workflow.python("validate", spec.name),
                cwd=repo,
                argv=(base,),
            )
            output.append(stderr)
            if stdout.strip() == "corpus-only":
                mode = "changed-corpus-toolchain-bump"
            else:
                mode = "full-toolchain-bump"
                break
    run.mode = mode
    auto = run.validate_roots_input == "auto"
    for shard in run.plan.matrix:
        roots = tuple(run.plan.shard_roots(shard, run.validate_roots_input).split())
        rules: set[str] = set()
        tests: set[str] = set()
        if mode == "full-toolchain-bump":
            for root in roots:
                found = [
                    path
                    for path in _find_regular_files(repo, root)
                    if path.endswith((".yaml", ".yml"))
                    and not fnmatch.fnmatchcase(path, "*/programs/*")
                ]
                for path in found:
                    if not path.endswith((".test.yaml", ".test.yml")):
                        _add_rulespec_target(repo, path, roots, auto, rules, tests)
                for path in found:
                    if path.endswith((".test.yaml", ".test.yml")):
                        _add_rulespec_target(repo, path, roots, auto, rules, tests)
        else:
            for path in run.changed:
                for root in roots:
                    if path.startswith(f"{root}/"):
                        _add_rulespec_target(repo, path, roots, auto, rules, tests)
        selection = ShardSelection(
            shard, roots, tuple(sorted(rules)), tuple(sorted(tests))
        )
        run.selections[shard] = selection
        prefix = f"[{shard}] " if len(run.plan.matrix) > 1 else ""
        if selection.rulespec_files or selection.test_files or not prefix:
            output.append(f"{prefix}RuleSpec validation mode: {mode}\n")
            output.extend(f"{prefix}- {path}\n" for path in selection.rulespec_files)
            output.extend(f"{prefix}- {path}\n" for path in selection.test_files)
    selected = sum(len(item.rulespec_files) for item in run.selections.values())
    output.append(
        f"RuleSpec validation mode: {mode}; {selected} RuleSpec file(s) selected "
        f"across {len(run.plan.matrix)} shard(s) ({run.plan.scope})\n"
    )
    return _result(
        spec, 0, "".join(output), ["selection", "--base-ref", base, "--mode", mode]
    )


def _add_rulespec_target(
    repo: Path,
    path: str,
    roots: Sequence[str],
    auto: bool,
    rules: set[str],
    tests: set[str],
) -> None:
    """Mirror the selection step's add_rulespec_target bash function."""

    if fnmatch.fnmatchcase(path, "*/programs/*") or fnmatch.fnmatchcase(
        path, "programs/*"
    ):
        return
    if auto and not any(
        path.startswith(f"{root}/{content}/")
        for root in roots
        for content in RULESPEC_CONTENT_ROOTS
    ):
        return

    def add(candidate: str, into: set[str]) -> None:
        if (repo / candidate).is_file():
            into.add(candidate)

    if path.endswith(".test.yaml"):
        add(path, tests)
        add(path.removesuffix(".test.yaml") + ".yaml", rules)
    elif path.endswith(".test.yml"):
        add(path, tests)
        add(path.removesuffix(".test.yml") + ".yml", rules)
    elif path.endswith(".yaml"):
        add(path, rules)
        add(path.removesuffix(".yaml") + ".test.yaml", tests)
    elif path.endswith(".yml"):
        add(path, rules)
        add(path.removesuffix(".yml") + ".test.yml", tests)


class _SkipList:
    """A step's skip list, built the first time a shard reaches it.

    The validate and proof steps run their SKIP script (modules with an active
    waiver) only after their no-files early exit, then append the
    retired-schema skip file. A failing script fails that shard's step and
    leaves an empty list behind.
    """

    def __init__(self, run: WorkflowRun, step: str) -> None:
        self.run = run
        self.step = step
        self._built: tuple[frozenset[str], str | None] | None = None

    def build(self) -> tuple[frozenset[str], str | None]:
        """Return (skip list, error) for a shard that reached the list."""

        if self._built is None:
            active: set[str] = set()
            error = None
            if (self.run.repo / "known-validation-gaps.yaml").is_file():
                code, stdout, stderr = _run_embedded_python(
                    self.run.workflow.python("validate", self.step), cwd=self.run.repo
                )
                if code:
                    error = (stderr or stdout).strip() or f"exit status {code}"
                else:
                    active = {line for line in stdout.splitlines() if line}
            skipped = (
                frozenset() if error else frozenset(active | self.run.retired_skip)
            )
            self._built = (skipped, error)
        return self._built


def _gate_validate(run: WorkflowRun, spec: GateSpec) -> GateResult:
    skip_list = _SkipList(run, spec.name)
    workers = _render_number(run.caller.inputs["validation-workers"])
    notes: dict[str, list[str]] = {}
    planned: list[tuple[str, CliInvocation]] = []
    outcomes: list[tuple[str, int, str]] = []
    for shard in run.plan.matrix:
        files = run.selections[shard].rulespec_files
        lines = notes.setdefault(shard, [])
        if not files:
            outcomes.append(
                (shard, 0, "No RuleSpec YAML files selected for validation.\n")
            )
            continue
        skipped, error = skip_list.build()
        run.validate_skip[shard] = skipped
        if error:
            outcomes.append((shard, 1, f"{error}\n"))
            continue
        selected = []
        for file in files:
            if file in skipped:
                lines.append(
                    f"SKIPPED (known-validation-gaps validate_failures): {file}"
                )
            else:
                selected.append(str(run.repo / file))
        if not selected:
            lines.append("All selected RuleSpec YAML files are skipped.")
            outcomes.append((shard, 0, "\n".join(lines) + "\n"))
            continue
        if re.fullmatch(r"[1-4]", workers) is None:
            outcomes.append(
                (shard, 1, "validation-workers must be an integer between 1 and 4\n")
            )
            continue
        count = min(int(workers), len(selected))
        for worker in range(count):
            planned.append(
                (
                    shard,
                    CliInvocation(
                        (
                            "validate",
                            *selected[worker::count],
                            "--skip-reviewers",
                            "--corpus-path",
                            str(run.paths["corpus"]),
                            "--axiom-rules-engine-path",
                            str(run.paths["engine"]),
                        ),
                        cwd=run.repo,
                        supervised=True,
                    ),
                )
            )
    results = _run_cli_batch(
        [invocation for _, invocation in planned],
        jobs=run.jobs,
        keyring=run.keyring,
        signing_roots=run.signing_roots,
    )
    by_shard: dict[str, list[tuple[int, str]]] = {}
    for (shard, _), outcome in zip(planned, results, strict=True):
        by_shard.setdefault(shard, []).append(outcome)
    for shard, chunk_outcomes in by_shard.items():
        prelude = "".join(f"{line}\n" for line in notes.get(shard, []))
        outcomes.append(
            (
                shard,
                1 if any(code for code, _ in chunk_outcomes) else 0,
                prelude + "".join(text for _, text in chunk_outcomes),
            )
        )
    return _aggregate(
        spec,
        run,
        outcomes,
        ["validate", "{selected files}", "--skip-reviewers"],
        note=f"validation-workers={workers}; direct subcommand under library-level release verification",
    )


def _gate_companion_tests(run: WorkflowRun, spec: GateSpec) -> GateResult:
    planned: list[tuple[str, CliInvocation]] = []
    messages: dict[str, list[str]] = {}
    for shard in run.plan.matrix:
        lines = messages.setdefault(shard, [])
        # The skip list exists only where the validate step wrote it.
        shard_skip = run.validate_skip.get(shard, frozenset())
        groups: dict[str, list[str]] = {}
        for test_file in run.selections[shard].test_files:
            module = test_file
            if test_file.endswith(".test.yaml"):
                module = test_file.removesuffix(".test.yaml") + ".yaml"
            elif test_file.endswith(".test.yml"):
                module = test_file.removesuffix(".test.yml") + ".yml"
            if module in shard_skip:
                lines.append(
                    "SKIPPED companion (known-validation-gaps validate_failures): "
                    f"{test_file}"
                )
                continue
            jurisdiction = test_file.split("/", 1)[0]
            relative = test_file.removeprefix(f"{jurisdiction}/")
            groups.setdefault(jurisdiction, []).append(relative)
        if not groups:
            lines.append("No RuleSpec companion tests selected; skipping.")
        for jurisdiction, files in groups.items():
            planned.append(
                (
                    shard,
                    CliInvocation(
                        (
                            "test",
                            "--root",
                            str(run.repo / jurisdiction),
                            "--axiom-rules-engine-path",
                            str(run.paths["engine"]),
                            *files,
                        ),
                        cwd=run.repo,
                        environment={
                            "AXIOM_RULESPEC_REPO_ROOTS": (
                                f"{run.repo}:{run.paths['rulespec_us']}"
                            )
                        },
                    ),
                )
            )
    results = _run_cli_batch(
        [invocation for _, invocation in planned],
        jobs=run.jobs,
        keyring=run.keyring,
        signing_roots=run.signing_roots,
    )
    by_shard: dict[str, list[tuple[int, str]]] = {
        shard: [] for shard in run.plan.matrix
    }
    for (shard, _), outcome in zip(planned, results, strict=True):
        by_shard[shard].append(outcome)
    outcomes = [
        (
            shard,
            1 if any(code for code, _ in by_shard[shard]) else 0,
            "".join(f"{line}\n" for line in messages[shard])
            + "".join(text for _, text in by_shard[shard]),
        )
        for shard in run.plan.matrix
    ]
    return _aggregate(spec, run, outcomes, ["test", "{selected tests}"])


def _gate_proof_validate(run: WorkflowRun, spec: GateSpec) -> GateResult:
    skip_list = _SkipList(run, spec.name)
    planned: list[tuple[str, CliInvocation]] = []
    outcomes: list[tuple[str, int, str]] = []
    notes: dict[str, list[str]] = {}
    for shard in run.plan.matrix:
        files = run.selections[shard].rulespec_files
        if not files:
            outcomes.append(
                (shard, 0, "No RuleSpec YAML files selected for proof validation.\n")
            )
            continue
        skipped, error = skip_list.build()
        if error:
            outcomes.append((shard, 1, f"{error}\n"))
            continue
        lines = notes.setdefault(shard, [])
        prefixed = []
        for file in files:
            if file in skipped:
                lines.append(
                    f"SKIPPED (known-validation-gaps validate_failures): {file}"
                )
            else:
                prefixed.append(str(run.repo / file))
        if not prefixed:
            lines.append("All selected RuleSpec YAML files are skipped.")
            outcomes.append((shard, 0, "\n".join(lines) + "\n"))
            continue
        planned.append(
            (
                shard,
                CliInvocation(
                    (
                        "proof-validate",
                        *prefixed,
                        "--corpus-path",
                        str(run.paths["corpus"]),
                    ),
                    cwd=run.repo,
                    supervised=True,
                ),
            )
        )
    results = _run_cli_batch(
        [invocation for _, invocation in planned],
        jobs=run.jobs,
        keyring=run.keyring,
        signing_roots=run.signing_roots,
    )
    for (shard, _), (code, text) in zip(planned, results, strict=True):
        prelude = "".join(f"{line}\n" for line in notes.get(shard, []))
        outcomes.append((shard, code, prelude + text))
    return _aggregate(
        spec,
        run,
        outcomes,
        ["proof-validate", "{selected files}"],
        note="direct subcommand under library-level release verification",
    )


def _gate_money_atoms(run: WorkflowRun, spec: GateSpec) -> GateResult:
    if not run.caller.inputs["run-money-atom-check"]:
        return _result(spec, 0, "Disabled by caller run-money-atom-check.\n", [])
    if run.validate_roots_input == "auto":
        atomic_roots = [
            f"{root}/{marker}"
            for root in run.plan.roots.split()
            for marker in RULESPEC_CONTENT_ROOTS
            if (run.repo / root / marker).is_dir()
        ]
    else:
        atomic_roots = run.plan.roots.split()
    files = [
        path
        for root in atomic_roots
        if (run.repo / root).is_dir()
        for path in _find_regular_files(run.repo, root)
        if path.endswith((".yaml", ".yml"))
        and not path.endswith((".test.yaml", ".test.yml"))
        and not fnmatch.fnmatchcase(path, "*/programs/*")
    ]
    if not files:
        return _result(
            spec, 0, "No RuleSpec YAML files found for money-atom check.\n", []
        )
    command = [
        "proof-validate",
        *files,
        "--money-atoms-only",
        "--corpus-path",
        str(run.paths["corpus"]),
    ]
    if (run.repo / "known-missing-money-atoms.yaml").is_file():
        command += ["--ratchet-file", "known-missing-money-atoms.yaml"]
    code, output = _run_cli(command, cwd=run.repo, supervised=True)
    return _result(
        spec,
        code,
        output,
        [*command[:1], f"{{{len(files)} files}}", *command[1 + len(files) :]],
        note=f"first shard ({run.plan.first}); direct subcommand under library-level release verification",
    )


def _gate_oracle_coverage(run: WorkflowRun, spec: GateSpec) -> GateResult:
    if run.mode != "full-toolchain-bump":
        return _result(
            spec, 0, "Full oracle coverage is not selected for changed-file mode.\n", []
        )
    command = [
        "oracle-coverage",
        "--root",
        str(run.repo),
        "--fail-on-unmapped",
        "--fail-on-untested-comparable",
        "--limit",
        "50",
    ]
    return _result(spec, *_run_cli(command, cwd=run.repo), command)


def _gate_changed_oracle_coverage(run: WorkflowRun, spec: GateSpec) -> GateResult:
    if run.mode == "full-toolchain-bump":
        return _result(spec, 0, "Not used for full-toolchain-bump mode.\n", [])
    command = ["oracle-coverage", "--root", str(run.repo), "--json"]
    note = (
        f"classifier executed from encode pin {run.caller.refs['encode']} in "
        f"{run.paths['encode']}"
    )
    shards = [
        shard for shard in run.plan.matrix if run.selections[shard].rulespec_files
    ]
    outcomes: list[tuple[str, int, str]] = [
        (shard, 0, "No changed RuleSpec YAML files selected for oracle coverage.\n")
        for shard in run.plan.matrix
        if shard not in shards
    ]
    if shards:
        code, stdout, stderr = _run_pinned_process(
            run.paths["encode"],
            run.caller.refs["encode"],
            command,
            stderr=subprocess.PIPE,
        )
        coverage = run.temp / "policyengine-oracle-coverage.json"
        coverage.write_text(stdout, encoding="utf-8")
        for shard in shards:
            if code:
                outcomes.append((shard, code, stdout + stderr))
                continue
            file_list = run.temp / f"rulespec-files-{run.plan.matrix.index(shard)}.txt"
            file_list.write_text(
                "".join(f"{path}\n" for path in run.selections[shard].rulespec_files),
                encoding="utf-8",
            )
            filter_code, filter_stdout, filter_stderr = _run_embedded_python(
                run.workflow.python(
                    "validate",
                    "Validate changed PolicyEngine oracle coverage classification",
                ),
                cwd=run.repo,
                argv=(str(coverage), str(file_list)),
                environment=run.simulation.github_environment(run.repo),
            )
            outcomes.append((shard, filter_code, filter_stdout + filter_stderr))
    order = {shard: index for index, shard in enumerate(run.plan.matrix)}
    outcomes.sort(key=lambda outcome: order[outcome[0]])
    return _aggregate(spec, run, outcomes, command, note=note)


def _gate_repository_tests(run: WorkflowRun, spec: GateSpec) -> GateResult:
    if not run.caller.inputs["run-pytest"]:
        return _result(spec, 0, "Disabled by caller run-pytest.\n", [])
    has_tests = any(
        fnmatch.fnmatchcase(PurePosixPath(path).name, "test_*.py")
        or fnmatch.fnmatchcase(PurePosixPath(path).name, "*_test.py")
        for path in _find_regular_files(run.repo, "tests")
    )
    if not ((run.repo / "tests").is_dir() and has_tests):
        return _result(spec, 0, "No Python tests found; skipping pytest.\n", [])
    command = [sys.executable, "-m", "pytest", "-q", "tests"]
    return _result(
        spec,
        *_run_process(command, run.repo, environment=_repository_test_environment()),
        command,
        note=f"first shard ({run.plan.first}); runs after every trusted gate, as in CI",
    )


def _caller_overrides(args: argparse.Namespace) -> CallerOverrides:
    return CallerOverrides(
        registry_url=getattr(args, "corpus_release_registry_url", None),
        registry_anon_key=getattr(args, "corpus_release_registry_anon_key", None),
    )


def _run_ci_in_checkout(
    args: argparse.Namespace,
    source: Path,
    repo: Path,
    mismatches: list[DependencyMismatch],
    notes: list[str],
    state: dict[str, Any],
    *,
    base_ref: str | None = None,
) -> tuple[list[GateResult], CallerConfig, ShardPlan | None, tuple[str, ...]]:
    """Resolve and run every gate against the committed checkout ``repo``.

    ``args.base_ref`` is already the base commit resolved in ``source``;
    ``base_ref`` is the name the caller gave. Dependency checkout defaults
    stay siblings of the ``source`` checkout.
    """

    base_ref = base_ref or args.base_ref

    plan: ShardPlan | None = None
    head = _git(repo, "rev-parse", "HEAD", check=False)
    state["head"] = head.stdout.strip() if head.returncode == 0 else ""
    caller = find_caller_workflow(repo, overrides=_caller_overrides(args))
    state["caller"] = caller
    if caller.workflow_sha not in SUPPORTED_WORKFLOW_PINS:
        raise _unsupported_pin_error(caller.workflow_sha, caller.path)
    pin = SUPPORTED_WORKFLOW_PINS[caller.workflow_sha]
    embedded = pin.gate_parameters.embedded_scripts
    jobs = getattr(args, "jobs", None)
    jobs = 1 if jobs is None else int(jobs)
    if jobs < 1:
        raise ValueError("--jobs must be at least 1")
    retired_keys = tuple(getattr(args, "corpus_release_retired_public_key", None) or ())
    if retired_keys and not pin.gate_parameters.retired_corpus_release_key:
        notes.append(
            f"validate-rulespec@{caller.workflow_sha[:8]} provisions only the "
            "current corpus release key; --corpus-release-retired-public-key "
            "is ignored"
        )
        retired_keys = ()
    verify_dependency_inputs(caller)
    simulation: PullRequestSimulation | None = None
    workflow: PinnedWorkflow | None = None
    if embedded:
        workflow = PinnedWorkflow(caller.workflow_sha)
        # Refs (the base branch name, origin) belong to the source checkout.
        simulation = simulate_pull_request(
            source, base_ref, getattr(args, "pull_request", None)
        )
        resolve_workflow_toolchain(workflow, repo, simulation)
    else:
        verify_toolchain_base_binding(repo, args.base_ref)
    toolchain = load_rulespec_toolchain(repo)
    verify_rulespec_validation_waiver_set(repo)
    paths = resolve_dependency_paths(args, source)
    release_cache = (
        f"releases/{toolchain.corpus_release}/"
        f"{toolchain.corpus_release_content_sha256}.json"
    )
    for name, path in paths.items():
        mismatch = verify_dependency_checkout(
            name,
            path,
            caller.refs[name],
            caller.path,
            allow_ref_mismatch=args.allow_ref_mismatch,
            ignored_untracked=(
                frozenset({release_cache}) if name == "corpus" else frozenset()
            ),
        )
        if mismatch:
            mismatches.append(mismatch)
    pinned_version = encoder_version_at_pin(paths["encode"], caller.refs["encode"])
    encoder_mismatch = verify_ambient_encoder(
        caller.refs["encode"],
        pinned_version,
        caller.path,
        allow_encoder_mismatch=getattr(args, "allow_encoder_mismatch", False),
    )
    if encoder_mismatch:
        mismatches.append(encoder_mismatch)
    python_mismatch = verify_python_version(
        str(caller.inputs.get("python-version", "3.14")),
        caller.path,
        allow_encoder_mismatch=getattr(args, "allow_encoder_mismatch", False),
    )
    if python_mismatch:
        mismatches.append(python_mismatch)
    if workflow is not None:
        release_path, release_note = acquire_workflow_release_object(
            workflow, toolchain, paths["corpus"], caller, offline=args.offline
        )
        notes.append(release_note)
    else:
        release_path = acquire_release_object(
            toolchain,
            paths["corpus"],
            caller.release_base_url,
            offline=args.offline,
        )
    authenticate_release_provenance(
        release_path,
        paths["corpus"],
        caller.refs["corpus"],
        caller.path,
    )
    validate_roots_input = args.roots or caller.validate_roots
    if workflow is not None and simulation is not None:
        plan = compute_shard_plan(
            workflow,
            repo,
            simulation,
            validate_roots_input,
            caller.guard_programs_root,
        )
        roots = tuple(plan.roots.split())
        if not simulation.base_is_ancestor:
            notes.append(
                f"HEAD does not contain {base_ref}; CI validates the "
                "pull request's merge commit, so rebase for exact parity"
            )
    else:
        roots = resolve_roots(repo, validate_roots_input)
        if caller.guard_programs_root:
            roots = (*roots, "programs")
    if not roots:
        raise ValueError("No validation roots resolved")
    signing_roots = {
        kind: value
        for kind, value in (
            ("apply", getattr(args, "apply_public_key", None)),
            ("eval", getattr(args, "eval_public_key", None)),
        )
        if value
    }
    if "apply" not in signing_roots:
        notes.append(
            "no --apply-public-key: signed encoder apply manifests cannot be "
            "verified, so a change that CI accepts on its manifest fails here"
        )
    with local_corpus_release_verification(
        args.corpus_release_public_key,
        retired_public_keys=retired_keys,
        apply_public_key=signing_roots.get("apply"),
        eval_public_key=signing_roots.get("eval"),
    ):
        # Authenticate the release signature and its complete artifact
        # inventory even when changed-file selection produces no later
        # corpus-consuming gate.
        load_rulespec_local_corpus_release(repo, paths["corpus"])
        if workflow is not None and simulation is not None and plan is not None:
            run = WorkflowRun(
                caller=caller,
                workflow=workflow,
                paths=paths,
                repo=repo,
                simulation=simulation,
                plan=plan,
                validate_roots_input=validate_roots_input,
                keyring=(args.corpus_release_public_key, *retired_keys),
                jobs=jobs,
                signing_roots=signing_roots,
            )
            results = execute_workflow_gates(run)
            notes.extend(run.notes)
        else:
            results = execute_gates(args, caller, paths, roots)
    return results, caller, plan, roots


def run_ci(args: argparse.Namespace) -> int:
    source = args.repo.expanduser().resolve()
    mismatches: list[DependencyMismatch] = []
    notes: list[str] = []
    caller: CallerConfig | None = None
    plan: ShardPlan | None = None
    state: dict[str, Any] = {}
    repo: Path | None = None
    try:
        # Refs resolve in the source checkout, where branches, upstreams and
        # remote-tracking refs live; the fresh checkout only sees commits.
        head = resolve_commit(source, "HEAD")
        base_ref = args.base_ref
        args.base_ref = resolve_commit(source, base_ref)
        with committed_checkout(source, head, args.base_ref) as repo:
            args.repo = repo
            note = uncommitted_changes_note(source)
            if note:
                notes.append(note)
            results, caller, plan, roots = _run_ci_in_checkout(
                args, source, repo, mismatches, notes, state, base_ref=base_ref
            )
        results = [_map_result_paths(result, repo, source) for result in results]
    except Exception as exc:
        message = str(exc)
        if repo is not None:
            message = message.replace(str(repo), str(source))
        if args.json:
            print(
                json.dumps(
                    {
                        "passed": False,
                        "verdict": "FAIL",
                        "dependency_mismatches": [asdict(item) for item in mismatches],
                        "resolutions": list(state["caller"].resolutions)
                        if "caller" in state
                        else [],
                        "notes": notes,
                        "resolution_error": message,
                        "gates": [],
                    },
                    indent=2,
                )
            )
        else:
            for mismatch in mismatches:
                print(f"WARNING: {mismatch.banner_line()}", file=sys.stderr)
            print(f"axiom-encode ci resolution failed: {message}", file=sys.stderr)
        return 1
    finally:
        args.repo = source
    gates_passed = all(result.status == "PASS" for result in results)
    verdict = ci_verdict(gates_passed, mismatches)
    if args.json:
        report: dict[str, Any] = {
            "passed": verdict == "PASS",
            "verdict": verdict,
            "caller": str(_source_path(caller.path, repo or source, source)),
            "validated_commit": state.get("head", ""),
            "workflow_sha": caller.workflow_sha,
            "dependency_mismatches": [asdict(item) for item in mismatches],
            "resolutions": list(caller.resolutions),
            "notes": notes,
            "roots": roots,
            "gates": [asdict(result) for result in results],
        }
        if plan is not None:
            report["shards"] = {
                "matrix": list(plan.matrix),
                "first": plan.first,
                "scope": plan.scope,
            }
        print(json.dumps(report, indent=2))
    else:
        for mismatch in mismatches:
            print(f"WARNING: {mismatch.banner_line()}", file=sys.stderr)
        print(
            f"validate-rulespec caller: {_source_path(caller.path, repo or source, source)} "
            f"@ {caller.workflow_sha}"
        )
        print(f"  validated commit: {state.get('head', '')}")
        for resolution in caller.resolutions:
            print(f"  resolved {resolution}")
        if plan is not None:
            print(f"  shards: {len(plan.matrix)} ({plan.scope}); first {plan.first}")
        for note in notes:
            print(f"  note: {note}")
        for result in results:
            print(f"{result.status:4} {result.name}")
            for failure in result.failures:
                print(f"     - {failure}")
            if result.note:
                print(f"     {result.note}")
        print(f"ci parity result: {verdict}")
        if mismatches:
            print("MISMATCHED DEPENDENCIES:")
            for mismatch in mismatches:
                print(f"  - {mismatch.banner_line()}")
    return (
        0 if verdict == "PASS" else 3 if verdict == "PASS-WITH-MISMATCHED-DEPS" else 1
    )


def _map_result_paths(result: GateResult, checkout: Path, source: Path) -> GateResult:
    """Report paths in the removed temporary checkout as ``source`` paths."""

    def remap(text: str) -> str:
        return text.replace(str(checkout), str(source))

    return GateResult(
        result.gate,
        result.name,
        result.status,
        [remap(item) for item in result.command],
        [remap(item) for item in result.failures],
        remap(result.output),
        remap(result.note) if result.note else result.note,
    )


def _source_path(path: Path, checkout: Path, source: Path) -> Path:
    """Map a path inside the committed checkout back to the source checkout."""

    try:
        return source / path.relative_to(checkout)
    except ValueError:
        return path


def ci_verdict(gates_passed: bool, mismatches: Sequence[DependencyMismatch]) -> str:
    if not gates_passed:
        return "FAIL"
    return "PASS-WITH-MISMATCHED-DEPS" if mismatches else "PASS"


def workflow_axiom_invocations(path: Path) -> list[tuple[str, tuple[str, ...]]]:
    """Extract axiom-encode invocations from every job's run blocks."""

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    invocations = []
    for job in payload["jobs"].values():
        for step in job.get("steps", []):
            run = step.get("run", "") if isinstance(step, dict) else ""
            logical_run = run.replace("\\\n", " ")
            # Only command positions count: a line start, or the operand after
            # the supervisor's `--`. Mentions in comments, backticks and
            # strings are not invocations.
            for match in re.finditer(
                r"(?m)(?:^[ \t]*|--[ \t]+)"
                r"(?:/opt/axiom-verification/)?axiom-encode\s+([a-z-]+)([^\n]*)",
                logical_run,
            ):
                command = match.group(1)
                try:
                    tokens = shlex.split(match.group(2))
                except ValueError:
                    tokens = match.group(2).split()
                flags = tuple(token for token in tokens if token.startswith("--"))
                if command in {
                    "validation-waivers",
                    "guard-generated",
                    "validate",
                    "test",
                    "proof-validate",
                    "oracle-coverage",
                }:
                    invocations.append((command, flags))
    return invocations


def workflow_steps(path: Path) -> list[tuple[str, dict[str, Any]]]:
    """Return every (job, step) of a pinned workflow, in file order."""

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return [
        (job_name, step)
        for job_name, job in payload["jobs"].items()
        for step in job.get("steps", [])
        if isinstance(step, dict)
    ]


def workflow_gate_coverage(path: Path) -> dict[str, tuple[str, ...]]:
    """Map every parity-relevant workflow step to its registry gate(s)."""

    names = {step.get("name") for _, step in workflow_steps(path)}
    return {
        name: WORKFLOW_GATE_STEPS[name] for name in names & WORKFLOW_GATE_STEPS.keys()
    }


def workflow_gate_contract(
    step: dict[str, Any], payload: dict[str, Any], job: str = "validate"
) -> dict[str, Any]:
    """Return behavior-bearing reusable-workflow context for one gate step."""

    trigger = payload.get("on", payload.get(True, {}))
    workflow_call = (
        trigger.get("workflow_call", {}) if isinstance(trigger, dict) else {}
    )
    raw_inputs = (
        workflow_call.get("inputs", {}) if isinstance(workflow_call, dict) else {}
    )
    inputs = {
        name: {
            "type": declaration.get("type"),
            "default": declaration.get("default"),
        }
        for name, declaration in sorted(raw_inputs.items())
        if isinstance(name, str) and isinstance(declaration, dict)
    }
    job_payload = payload.get("jobs", {}).get(job, {})
    contract = {
        "step": step,
        "workflow_call_inputs": inputs,
        "workflow_env": payload.get("env"),
        "validate_job_env": job_payload.get("env"),
        "workflow_defaults": payload.get("defaults"),
        "validate_job_defaults": job_payload.get("defaults"),
    }
    if job != "validate":
        contract["job"] = job
    return contract
