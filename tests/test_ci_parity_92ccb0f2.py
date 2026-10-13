"""Parity tests for validate-rulespec 92ccb0f2, rulespec-us's pin since 4c9aeadd.

92ccb0f2 differs from 6f11be26 in four ways ci reproduces: the caller's
resolver also reads axiom_compose_ref; the pre-check shape-checks ProgramSpecs
and reads guard-programs-root; dependency checkouts move out of the rules
checkout to $AXIOM_DEPENDENCY_ROOT; and a pinned composer runtime is built and
passed to the waiver audit, validation and companion tests.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import yaml
from hypothesis import given, settings
from hypothesis import strategies as st

from axiom_encode import ci_parity
from axiom_encode.ci_parity import (
    SUPPORTED_WORKFLOW_PINS,
    WORKFLOW_DIRECTORY,
    WORKFLOW_ENVIRONMENT_STEPS,
    WORKFLOW_GATE_STEPS,
    WORKFLOW_RESOLUTION_STEPS,
    DependencyLayout,
    PinnedWorkflow,
    isolated_dependency_layout,
    parse_caller_workflow,
    program_spec_parser_mismatch,
    run_ci,
    validate_composer_pin,
    workflow_gate_contract,
    workflow_steps,
)
from tests import test_ci_parity_workflow_gates as gates

PIN_6F11BE26 = "6f11be2655f79dd0a3b582db46525f58332ca120"
PIN_92CCB0F2 = "92ccb0f2266bf06aba6111d2bfd0e5f03af55fc3"
COMPOSE_REF = "fabe0b3b3fd6e90d3e8f075516f9b668f524f711"
FIXTURES = Path(__file__).parent / "fixtures" / "ci_parity"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

# The contract of every step of every job, environment steps included, so a
# path-only change like .github#120's cannot slip past review.
STEP_CONTRACT_SHA256_92CCB0F2: dict[tuple[str, str], str] = {
    (
        "shards",
        "Checkout rules repository",
    ): "0892b1a54ae0b94fd847383f2c6d0f6772e826854b2532a18a8fd3e781e5545b",
    (
        "shards",
        "Reject unsupported tracked paths",
    ): "ab21ce2cbcc867f9c0cfb4b0eddd5b488da5c63b0df7d0cc8ce8ef4646c3b2f9",
    (
        "shards",
        "Set up Python for the RuleSpec pre-check",
    ): "0bcb859068d25c8310ed9f1936d422defcc05294fd83031fd4cb91f114c21167",
    (
        "shards",
        "Install the ProgramSpec shape-check parser",
    ): "562cebb844c7c7425b35dfb3a48ab477adab2b8c1d464a53924645a4e56545ff",
    (
        "shards",
        "Reject unmanifested RuleSpec content",
    ): "b13476d67d49b0061b772800441f2f49801f3a87d9dfcc84bf1ec7cc4ab2eaca",
    (
        "shards",
        "Compute validation shards",
    ): "fc06d31869a83a5af54cacb79da692cb1552b67d647c2ae9be8b4d98ab4dfcff",
    (
        "validate",
        "Free runner disk space",
    ): "9255b47b445ace4abbd83215562e752c2c8ab2095f5de312fc74a3fe6f8832fe",
    (
        "validate",
        "Checkout rules repository",
    ): "7036bc57bb60a420b2e497bc67f474bc1e3e54b188f5d9cc6030f37fff60ceda",
    (
        "validate",
        "Authorize exact reviewed migration",
    ): "b5fab18e0100f8e6b4010f789167ff7753deb9b8a9fb5e5fd5462efd9d067355",
    (
        "validate",
        "Resolve shard validation roots",
    ): "b9c80a39aea0ba98885c1190970e34dba88697eaf267dd5e0101f12bcb0374a5",
    (
        "validate",
        "Set up Python",
    ): "19f2140f49a826bcee34567186bdb6c9e1f7931571f6dfdf4a305e39443a3fa0",
    (
        "validate",
        "Resolve RuleSpec toolchain",
    ): "7a4dc0d78cdc68896941e5a594c411906f06dec5062e3f790eeefb470d3f4259",
    (
        "validate",
        "Validate immutable dependency inputs",
    ): "25a5ea32893fc86528c701cf27b73ab12deba8016566d0b9d8ccfd77a611032e",
    (
        "validate",
        "Reserve isolated dependency directories",
    ): "114d8c9e9a9c79191625efce8c1f612bc51e990631f5bf4ee3a98aaa97bdf7a2",
    (
        "validate",
        "Checkout axiom-encode",
    ): "4ac350a03eed278c60a7a6668f2f83fbfffebef504b32017244bc464bc200ddf",
    (
        "validate",
        "Relocate dependency axiom-encode",
    ): "22165f9bd2062716bfe5418ec50c860493c241b20037bb882a16e189a1a6b744",
    (
        "validate",
        "Checkout axiom-rules-engine",
    ): "634f0b29cb55fdd457b1d2691b0798aadfa052be037ae83b8c50c1b4bcc9a5e8",
    (
        "validate",
        "Relocate dependency axiom-rules-engine",
    ): "486125a78d90dbfe5df2a941bf445f8e80573d8bf354224618830d9f0ea80ad7",
    (
        "validate",
        "Checkout axiom-corpus",
    ): "8b7b43871ee43e86099538d6ac3aa07235fe60ed9b2fb55b037fc7df30ecfb02",
    (
        "validate",
        "Relocate dependency axiom-corpus",
    ): "5acda3b79511b4578171f9cd77ac45ab9efe7146159d83547d861bdc2b41f6ff",
    (
        "validate",
        "Checkout rulespec-us canonical targets",
    ): "ea40a33fb31bca58ff8033a063cb2f990a623b868cf474f0d646f4686c645cd4",
    (
        "validate",
        "Relocate dependency rulespec-us",
    ): "612710bcdee509d2e8f99085a95d2c7550da62ea68348f1a723772ac4a2e4ef5",
    (
        "validate",
        "Authenticate dependency commits",
    ): "18086035206fb0503909de64b8d69416eaa00a911f5025fb0ff2ff10b72bdf61",
    (
        "validate",
        "Validate composer pin",
    ): "4ba5ac7c7bde4a9f5df521ae51df9d2760626ca42110e1ce529ff17dacac8ce7",
    (
        "validate",
        "Checkout axiom-compose",
    ): "f325e48048045f3ecb43e4f36c1d949d839ab80017d83b957e93b10001eace21",
    (
        "validate",
        "Relocate dependency axiom-compose",
    ): "3d72f029262da06b640639ebc8239d3d4f22577485fe927f526d3fa73a12ba4b",
    (
        "validate",
        "Authenticate composer commit",
    ): "ab03a8149bbb57c5775e1b27192e14204097b46a976d5c5d3eb350c14ee8bb39",
    (
        "validate",
        "Install uv for composer runtime",
    ): "ccd138a76f1805a3c8d37b5c885f7b7831ab79923ff38f12cbbb3ec6d05ee827",
    (
        "validate",
        "Install Rust toolchain",
    ): "fe048de8709513b3fdb33086730caed623750f6a0106981edb8e9c1c4909f3de",
    (
        "validate",
        "Set up Go 1.26.1",
    ): "b1c4733e10b56244128d2bf6cb10e7589f2184dcec63d8ad7672737fb22eaced",
    (
        "validate",
        "Install Python dependencies",
    ): "1cc8a6f4914dd52ca4f74102f32c9a770adfc017abb5c3b22c036f10b44a81e4",
    (
        "validate",
        "Verify immutable retired-schema freeze",
    ): "554ff05c5a51c170ef836b45d757afd40f6c4b0a1fbe47e4b41a0a659c2abe01",
    (
        "validate",
        "Fetch pinned signed corpus release object",
    ): "a68c6748b9954d7e6203734e53608aab7d67e469b82f9a4b77ce9e57769648f0",
    (
        "validate",
        "Authenticate signed corpus provenance commit",
    ): "7d696d9c51afaf528f6855ace54841d426dc537faba0c9f8a120daa880202140",
    (
        "validate",
        "Provision protected verification supervisor",
    ): "d976d9ab296fdae0194e5c5945561ca4df5b20d55efe508257aae1e5f2e69e9b",
    (
        "validate",
        "Build protected axiom-compose runtime",
    ): "0fc9a5763cf668b05a7cebbdb62cbb89735d63e4b82140ad59f149cc3151376d",
    (
        "validate",
        "Build axiom-rules-engine",
    ): "2326314797d47046ddc59bbfee4b01fe3e9c10f1aa0a8a6724059755d2bc4e15",
    (
        "validate",
        "Expose axiom-rules-engine on PATH",
    ): "6d474b4ca11c9759ee3e9e7158c759bd6d3425aa56c786ed770d103b22d645fe",
    (
        "validate",
        "Link axiom-rules-engine as sibling checkout",
    ): "85efef68316daa2e6a596e02d0050c717fb379dd41f93c0998840222ec3199ed",
    (
        "validate",
        "Reject obsolete generated files",
    ): "9c3fa59a5449778a0607281c2fa1c4d1b34598ce31d0ccdeb9b6535dcf9a1036",
    (
        "validate",
        "Reject disallowed repository layout",
    ): "48ee16dcfd51f70d2751c63098508bb28a926d5731e09088cbdca71cefebe091",
    (
        "validate",
        "Enforce validation waiver ratchet",
    ): "c3888a0ca9828636fb08c200048d547ac19fc2547e98515c154f3c8d298514d8",
    (
        "validate",
        "Reject manual RuleSpec changes",
    ): "35e62793fce8542a21bda877fdc7ee609ac8d2f6e15e2adc167b08932bd73bfd",
    (
        "validate",
        "Select RuleSpec validation targets",
    ): "84e36ebc4ec25dee93625aa19c6e26bd2056f2de0fdc8aa17770fa14f6d89624",
    (
        "validate",
        "Validate RuleSpec YAML",
    ): "f9812918dca055703697aa43c8bf7c3b557418ede1b25ac12f41436e2cf838e8",
    (
        "validate",
        "Execute RuleSpec companion tests",
    ): "ad13a3aef68874ff6d57274ef2fc3f38c0832c6e07454414e343bc6fb80e0cbe",
    (
        "validate",
        "Validate RuleSpec proofs and claims",
    ): "4783087737a247cc28fd139ba7574371aab6e689ed18d6f2ff2ef8f8a9626b89",
    (
        "validate",
        "Require money proof atoms",
    ): "1b3af2e020474228799a03d9e4fe07652268ae50ad8c5547acfeb892c6df6f8c",
    (
        "validate",
        "Validate PolicyEngine oracle coverage classification",
    ): "08ae8fbbba35c381b08ca2d267fe49db9cf384b52798ce2d05258f367c88f667",
    (
        "validate",
        "Checkout changed-file oracle coverage classifier",
    ): "b55e9ef2a1246a84562a204b8d290b5c1d361b6c4ab38723e3fc9e1f27d1920f",
    (
        "validate",
        "Relocate dependency axiom-encode-oracle-coverage",
    ): "0ab604fc57faf3e56edf26bb40ddbf348b8e31d5754e0cc8e9c15de42050f26d",
    (
        "validate",
        "Install changed-file oracle coverage classifier",
    ): "026c9bf43c405c0cb655433afc5a6a4076ec96f990b54e5eb9146d2560d48987",
    (
        "validate",
        "Validate changed PolicyEngine oracle coverage classification",
    ): "0020f9c8d8a0783b5a0d64579468436164f41c4c18cd6e98365d93ef816c9e62",
    (
        "validate",
        "Run repository tests",
    ): "9f79c04c1a6deff195d11c94f243fcf4655f402410ebc717b84a459cba53e26e",
    (
        "validate-complete",
        "Check validation matrix result",
    ): "7881b8c4208283fff7af19072159352cde22a12daefafbcc92f93f3579095c88",
}
# Job-level keys outside the step contracts: time limits and the matrix
# bounds (.github#116) change no verdict ci reaches; see docs/ci-parity.md.
JOB_SETTINGS_92CCB0F2 = {
    "shards": {"name": "shards", "runs-on": "ubuntu-latest", "timeout-minutes": 10},
    "validate": {
        "name": "validate",
        "needs": "shards",
        "runs-on": "ubuntu-latest",
        "timeout-minutes": 180,
        "strategy": {
            "fail-fast": False,
            "max-parallel": 20,
            "matrix": {"shard": "${{ fromJSON(needs.shards.outputs.matrix) }}"},
        },
    },
    "validate-complete": {
        "name": "validate",
        "needs": "validate",
        "if": "${{ always() }}",
        "runs-on": "ubuntu-latest",
        "timeout-minutes": 5,
    },
}


def _workflow_payload(sha: str) -> dict[str, Any]:
    pin = SUPPORTED_WORKFLOW_PINS[sha]
    return yaml.safe_load((WORKFLOW_DIRECTORY / pin.fixture).read_text())


def _steps(payload: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    return {
        (job, step["name"]): step
        for job, body in payload["jobs"].items()
        for step in body.get("steps", [])
    }


# ---------------------------------------------------------------------------
# The packaged pin
# ---------------------------------------------------------------------------


def test_every_92ccb0f2_step_contract_is_pinned() -> None:
    fixture = WORKFLOW_DIRECTORY / SUPPORTED_WORKFLOW_PINS[PIN_92CCB0F2].fixture
    payload = yaml.safe_load(fixture.read_text())
    actual = {
        (job, step["name"]): hashlib.sha256(
            yaml.safe_dump(
                workflow_gate_contract(step, payload, job), sort_keys=True
            ).encode()
        ).hexdigest()
        for job, step in workflow_steps(fixture)
    }

    assert actual == STEP_CONTRACT_SHA256_92CCB0F2
    for _, name in actual:
        assert (
            (name in WORKFLOW_GATE_STEPS)
            + (name in WORKFLOW_RESOLUTION_STEPS)
            + (name in WORKFLOW_ENVIRONMENT_STEPS)
        ) == 1, name


def test_92ccb0f2_job_settings_are_reviewed() -> None:
    payload = _workflow_payload(PIN_92CCB0F2)
    settings_by_job = {
        job: {
            key: value for key, value in body.items() if key not in {"steps", "outputs"}
        }
        for job, body in payload["jobs"].items()
    }

    assert settings_by_job == JOB_SETTINGS_92CCB0F2
    assert (
        payload["jobs"]["shards"]["outputs"]
        == (_workflow_payload(PIN_6F11BE26)["jobs"]["shards"]["outputs"])
    )


_DEPENDENCY_PREFIXES = (
    "${{ github.workspace }}/_axiom/",
    "$GITHUB_WORKSPACE/_axiom/",
    "${{ env.AXIOM_DEPENDENCY_ROOT }}/",
    "${{ env.AXIOM_DEPENDENCY_STAGE }}/",
    "${AXIOM_DEPENDENCY_ROOT}/",
    "$AXIOM_DEPENDENCY_ROOT/",
    "./_axiom/",
    "_axiom/",
)


def _normalize_dependency_paths(text: str) -> str:
    """Map 6f11be26's _axiom paths and 92ccb0f2's dependency root to one token.

    Quoting around the path is normalized too: 92ccb0f2 quotes the expanded
    root where 6f11be26's relative _axiom paths needed no quotes.
    """

    for prefix in _DEPENDENCY_PREFIXES:
        text = text.replace(prefix, "@DEPENDENCY/")
    return re.sub(r'"(@DEPENDENCY/[^"\s]*)"', r"\1", text)


def _normalized(value: Any) -> Any:
    if isinstance(value, str):
        return _normalize_dependency_paths(value)
    if isinstance(value, dict):
        return {key: _normalized(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_normalized(item) for item in value]
    return value


def test_92ccb0f2_differs_from_6f11be26_only_as_reviewed() -> None:
    old_payload = _workflow_payload(PIN_6F11BE26)
    new_payload = _workflow_payload(PIN_92CCB0F2)
    old, new = _steps(old_payload), _steps(new_payload)

    assert set(old) <= set(new)
    assert set(new) - set(old) == {
        ("shards", "Set up Python for the RuleSpec pre-check"),
        ("shards", "Install the ProgramSpec shape-check parser"),
        ("validate", "Reserve isolated dependency directories"),
        ("validate", "Relocate dependency axiom-encode"),
        ("validate", "Relocate dependency axiom-rules-engine"),
        ("validate", "Relocate dependency axiom-corpus"),
        ("validate", "Relocate dependency rulespec-us"),
        ("validate", "Validate composer pin"),
        ("validate", "Checkout axiom-compose"),
        ("validate", "Relocate dependency axiom-compose"),
        ("validate", "Authenticate composer commit"),
        ("validate", "Install uv for composer runtime"),
        ("validate", "Build protected axiom-compose runtime"),
        ("validate", "Relocate dependency axiom-encode-oracle-coverage"),
    }
    changed = {key for key in old if old[key] != new[key]}
    path_only = {
        key for key in changed if _normalized(old[key]) == _normalized(new[key])
    }
    assert path_only == {
        ("validate", "Checkout axiom-encode"),
        ("validate", "Checkout axiom-rules-engine"),
        ("validate", "Checkout axiom-corpus"),
        ("validate", "Checkout rulespec-us canonical targets"),
        ("validate", "Authenticate dependency commits"),
        ("validate", "Install Python dependencies"),
        ("validate", "Fetch pinned signed corpus release object"),
        ("validate", "Authenticate signed corpus provenance commit"),
        ("validate", "Provision protected verification supervisor"),
        ("validate", "Build axiom-rules-engine"),
        ("validate", "Expose axiom-rules-engine on PATH"),
        ("validate", "Link axiom-rules-engine as sibling checkout"),
        ("validate", "Reject manual RuleSpec changes"),
        ("validate", "Validate RuleSpec proofs and claims"),
        ("validate", "Require money proof atoms"),
        ("validate", "Checkout changed-file oracle coverage classifier"),
    }
    semantic = changed - path_only
    assert semantic == {
        ("shards", "Reject unmanifested RuleSpec content"),
        ("validate", "Enforce validation waiver ratchet"),
        ("validate", "Validate RuleSpec YAML"),
        ("validate", "Execute RuleSpec companion tests"),
        ("validate", "Install changed-file oracle coverage classifier"),
    }
    # The classifier install reads its pyproject from the dependency root;
    # ci reads the same pinned file from the --encode-path checkout.
    classifier = ("validate", "Install changed-file oracle coverage classifier")
    old_install = _normalized(old[classifier])["run"].replace(
        "pathlib.Path(@DEPENDENCY/axiom-encode-oracle-coverage/pyproject.toml)",
        "PYPROJECT",
    )
    new_install = (
        _normalized(new[classifier])["run"]
        .replace("import os\n", "", 1)
        .replace(
            '(pathlib.Path(os.environ["AXIOM_DEPENDENCY_ROOT"]) / '
            '"axiom-encode-oracle-coverage/pyproject.toml")',
            "PYPROJECT",
        )
    )
    assert old_install == new_install
    precheck = new[("shards", "Reject unmanifested RuleSpec content")]
    assert set(precheck["env"]) - set(
        old[("shards", "Reject unmanifested RuleSpec content")]["env"]
    ) == {"GUARD_PROGRAMS_ROOT"}
    timeouts = {
        "Enforce validation waiver ratchet": 150,
        "Validate RuleSpec YAML": 120,
        "Execute RuleSpec companion tests": 60,
    }
    for name, minutes in timeouts.items():
        step = new[("validate", name)]
        assert step["timeout-minutes"] == minutes
        assert step["env"]["AXIOM_COMPOSE_REF"] == "${{ inputs.axiom-compose-ref }}"
        run = step["run"]
        assert run.count('"${composer_args[@]}" "${dependency_args[@]}"') == 1, name
        assert (
            'if [ -n "$AXIOM_COMPOSE_REF" ] && [ "$(basename "$GITHUB_WORKSPACE")" '
            '!= "rulespec-us" ]; then\n'
            '  dependency_args=(--rulespec-dependency-root "$AXIOM_DEPENDENCY_ROOT/'
            'rulespec-us")'
        ) in run, name
    audit = new[("validate", "Enforce validation waiver ratchet")]
    assert (
        'validation-waivers audit "${composer_args[@]}" "${dependency_args[@]}" '
        "\\\n  --json \\\n" in audit["run"]
    )
    # The supervisor drops this variable (parentOnlyEnvironmentNames), so the
    # effective audit fan-out stays ci_parity.CI_WAIVER_AUDIT_WORKERS.
    assert audit["env"]["AXIOM_ENCODE_WAIVER_AUDIT_WORKERS"] == (
        "${{ github.repository == 'TheAxiomFoundation/rulespec-us' && "
        "matrix.shard == 'us-wa' && '2' || '1' }}"
    )


def test_92ccb0f2_inputs_add_only_the_optional_composer_ref() -> None:
    old = _workflow_payload(PIN_6F11BE26)[True]["workflow_call"]["inputs"]
    new = _workflow_payload(PIN_92CCB0F2)[True]["workflow_call"]["inputs"]

    assert list(new) == [
        *list(old)[:2],
        "axiom-compose-ref",
        *list(old)[2:],
    ]
    assert new["axiom-compose-ref"] == {
        "description": (
            "Optional exact composer commit; requires a composer-aware encoder "
            "and matching workflow toolchain pin."
        ),
        "required": False,
        "type": "string",
        "default": "",
    }
    for name, declaration in old.items():
        assert {
            key: value for key, value in new[name].items() if key != "description"
        } == {
            key: value for key, value in declaration.items() if key != "description"
        }, name


def test_ci_reads_every_script_92ccb0f2_adds() -> None:
    workflow = PinnedWorkflow(PIN_92CCB0F2)

    layout = workflow.python("validate", "Reserve isolated dependency directories")
    composer = workflow.python("validate", "Validate composer pin")

    assert "dependency directory must be outside the RuleSpec checkout" in layout
    assert "composer input must match workflow-toolchain.toml" in composer


# ---------------------------------------------------------------------------
# The caller
# ---------------------------------------------------------------------------


def _caller_checkout(tmp_path: Path, caller_text: str, toolchain: str) -> Path:
    repo = tmp_path / "rulespec-us"
    (repo / ".github" / "workflows").mkdir(parents=True)
    (repo / ".axiom").mkdir()
    (repo / ".github" / "workflows" / "repository-checks.yml").write_text(caller_text)
    (repo / ".axiom" / "workflow-toolchain.toml").write_text(toolchain)
    return repo


def _real_caller() -> str:
    return (FIXTURES / "us-caller-92ccb0f2.yml").read_text()


def _real_toolchain() -> str:
    return (FIXTURES / "us-workflow-toolchain-92ccb0f2.toml").read_text()


def test_rulespec_us_92ccb0f2_caller_resolves_the_composer_ref(tmp_path: Path) -> None:
    repo = _caller_checkout(tmp_path, _real_caller(), _real_toolchain())
    os.environ.setdefault("NEXT_PUBLIC_SUPABASE_ANON_KEY", "anon")

    caller = parse_caller_workflow(
        repo / ".github" / "workflows" / "repository-checks.yml"
    )

    assert caller.workflow_sha == PIN_92CCB0F2
    assert caller.refs == {
        "encode": "3bf5a2afbff0391261ad65bd7497f1b0efaaff85",
        "engine": "89571cc2a938707fd60a5489b951135345697296",
        "corpus": "8f7d60aaced28ee4252b9237f9d6e02360dc34bc",
        "rulespec_us": "38d5c8c516f9243cedf8e07a22f96dde9ac66fe3",
        "compose": COMPOSE_REF,
    }
    assert caller.inputs["axiom-compose-ref"] == COMPOSE_REF
    assert caller.inputs["validate-roots"] == "auto"
    assert caller.inputs["validation-workers"] == 4
    assert caller.inputs["guard-programs-root"] is False
    assert caller.inputs["allow-retired-schema-prefreeze"] is True
    assert (
        "axiom-compose-ref: ${{ needs.workflow-toolchain.outputs.axiom_compose_ref }}"
        f" -> {COMPOSE_REF}"
    ) in caller.resolutions


def test_caller_resolver_digest_is_the_reviewed_compose_aware_one() -> None:
    caller = yaml.safe_load(_real_caller())
    run = caller["jobs"]["workflow-toolchain"]["steps"][1]["run"]

    digest = hashlib.sha256(run.encode("utf-8")).hexdigest()

    assert digest in ci_parity.RECOGNIZED_WORKFLOW_TOOLCHAIN_RESOLVERS
    assert '"axiom_compose_ref",' in run


def _literal_compose_caller(value: str) -> str:
    caller = yaml.safe_load(_real_caller())
    validate = caller["jobs"]["validate"]
    validate["with"]["axiom-compose-ref"] = value
    return yaml.safe_dump(caller, sort_keys=False)


def test_empty_composer_ref_skips_every_composer_step(tmp_path: Path) -> None:
    repo = _caller_checkout(tmp_path, _literal_compose_caller(""), _real_toolchain())
    os.environ.setdefault("NEXT_PUBLIC_SUPABASE_ANON_KEY", "anon")

    caller = parse_caller_workflow(
        repo / ".github" / "workflows" / "repository-checks.yml"
    )

    assert caller.inputs["axiom-compose-ref"] == ""
    assert "compose" not in caller.refs


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ("FABE0B3B3FD6E90D3E8F075516F9B668F524F711", "full lowercase commit SHA"),
        ("main", "full lowercase commit SHA"),
        ("1" * 40, "composer input must match workflow-toolchain.toml"),
    ],
    ids=["uppercase", "branch", "other-sha"],
)
def test_composer_ref_is_validated_where_ci_validates_it(
    tmp_path: Path, value: str, message: str
) -> None:
    repo = _caller_checkout(tmp_path, _literal_compose_caller(value), _real_toolchain())
    os.environ.setdefault("NEXT_PUBLIC_SUPABASE_ANON_KEY", "anon")

    # The caller parses: CI accepts any string for the input itself.
    caller = parse_caller_workflow(
        repo / ".github" / "workflows" / "repository-checks.yml"
    )
    assert caller.refs["compose"] == value

    with pytest.raises(ValueError, match=message):
        validate_composer_pin(PinnedWorkflow(PIN_92CCB0F2), repo, value)


def test_composer_pin_refuses_a_symlinked_workflow_toolchain(tmp_path: Path) -> None:
    repo = _caller_checkout(tmp_path, _real_caller(), _real_toolchain())
    toolchain = repo / ".axiom" / "workflow-toolchain.toml"
    real = tmp_path / "elsewhere.toml"
    real.write_text(toolchain.read_text())
    toolchain.unlink()
    toolchain.symlink_to(real)

    with pytest.raises(ValueError, match="a regular workflow-toolchain.toml"):
        validate_composer_pin(PinnedWorkflow(PIN_92CCB0F2), repo, COMPOSE_REF)


def test_matching_composer_pin_passes(tmp_path: Path) -> None:
    repo = _caller_checkout(tmp_path, _real_caller(), _real_toolchain())

    validate_composer_pin(PinnedWorkflow(PIN_92CCB0F2), repo, COMPOSE_REF)


# ---------------------------------------------------------------------------
# The pre-check: ProgramSpec shape and guard-programs-root
# ---------------------------------------------------------------------------

PROGRAM_SPEC = (
    "program: us/snap_benefit\n"
    "period: 2026\n"
    "outputs:\n"
    "  - snap_allotment\n"
    "scope:\n"
    "  federal:\n"
    "    - statutes/7/2017\n"
)
HAND_WRITTEN = "module:\n  id: hand-written\nformula: income * 0.3\n"


def _precheck(run: Any, sha: str) -> Any:
    return ci_parity._gate_unmanifested_rulespec(
        run, gates._spec("unmanifested_rulespec", sha)
    )


def _program_repo(tmp_path: Path, program: str) -> Path:
    repo = gates._repo(
        tmp_path,
        {"README.md": "readme\n", "us/statutes/7/2017.yaml": gates.PLAIN},
        name="rulespec-us",
        remote=gates.RULESPEC_US_REMOTE,
    )
    gates._commit(repo, {"programs/us/snap.yaml": program})
    return repo


def test_92ccb0f2_precheck_accepts_a_programspec_that_wires_encoded_modules(
    tmp_path: Path,
) -> None:
    repo = _program_repo(tmp_path, PROGRAM_SPEC)
    run = gates._run(repo, tmp_path, sha=PIN_92CCB0F2, plan=gates._plan("us"))

    result = _precheck(run, PIN_92CCB0F2)

    assert result.status == "PASS", result.output
    assert "1 changed ProgramSpec file(s) have the ProgramSpec shape" in result.output


def test_6f11be26_precheck_demands_a_manifest_for_the_same_programspec(
    tmp_path: Path,
) -> None:
    # The differential: 6f11be26 guards programs/ as a manifest root.
    repo = _program_repo(tmp_path, PROGRAM_SPEC)
    run = gates._run(repo, tmp_path, sha=PIN_6F11BE26, plan=gates._plan("us"))

    result = _precheck(run, PIN_6F11BE26)

    assert result.status == "FAIL"
    assert "programs/us/snap.yaml has no tracked encoder apply manifest" in (
        result.output
    )


def test_92ccb0f2_precheck_rejects_a_hand_written_module_under_programs(
    tmp_path: Path,
) -> None:
    repo = _program_repo(tmp_path, HAND_WRITTEN)
    run = gates._run(repo, tmp_path, sha=PIN_92CCB0F2, plan=gates._plan("us"))

    result = _precheck(run, PIN_92CCB0F2)

    assert result.status == "FAIL"
    assert "Changed programs/ files that are not ProgramSpecs" in result.output
    assert "programs/us/snap.yaml lacks ProgramSpec keys: outputs, period, program" in (
        result.output
    )


def test_92ccb0f2_precheck_reads_guard_programs_root(tmp_path: Path) -> None:
    # With the opt-in, programs/ also takes the manifest check, so the same
    # well-shaped ProgramSpec now fails; without GUARD_PROGRAMS_ROOT reaching
    # the script it would pass.
    repo = _program_repo(tmp_path, PROGRAM_SPEC)
    run = gates._run(
        repo,
        tmp_path,
        sha=PIN_92CCB0F2,
        inputs={"guard-programs-root": True},
        plan=gates._plan("us"),
    )

    result = _precheck(run, PIN_92CCB0F2)

    assert result.status == "FAIL"
    assert "programs/us/snap.yaml has no tracked encoder apply manifest" in (
        result.output
    )


@pytest.mark.parametrize("sha", [PIN_6F11BE26, PIN_92CCB0F2], ids=lambda s: s[:8])
@pytest.mark.parametrize("guard", [False, True])
def test_precheck_passes_guard_programs_root_only_where_the_step_declares_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sha: str, guard: bool
) -> None:
    seen: list[dict[str, str]] = []
    monkeypatch.setattr(
        ci_parity,
        "_run_embedded_python",
        lambda script, *, cwd, argv=(), environment=None, stdin=b"": (
            seen.append(dict(environment or {})) or (0, "", "")
        ),
    )
    repo = gates._repo(tmp_path, {"README.md": "x\n"})
    run = gates._run(
        repo,
        tmp_path,
        sha=sha,
        inputs={"guard-programs-root": guard},
        plan=gates._plan("us"),
    )

    _precheck(run, sha)

    expected = ("true" if guard else "false") if sha == PIN_92CCB0F2 else None
    assert seen[0].get("GUARD_PROGRAMS_ROOT") == expected


def test_program_spec_parser_is_the_pinned_pyyaml(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workflow = PinnedWorkflow(PIN_92CCB0F2)

    monkeypatch.setattr(yaml, "__version__", "6.0.3")
    assert program_spec_parser_mismatch(workflow) is None
    monkeypatch.setattr(yaml, "__version__", "6.0.2")
    mismatch = program_spec_parser_mismatch(workflow)
    assert mismatch is not None
    assert mismatch.banner_line() == (
        "program-spec-parser: HEAD pyyaml 6.0.2 != pinned pyyaml 6.0.3"
    )
    # 6f11be26 installs no parser of its own.
    assert program_spec_parser_mismatch(PinnedWorkflow(PIN_6F11BE26)) is None


# ---------------------------------------------------------------------------
# The isolated dependency layout
# ---------------------------------------------------------------------------


def test_isolated_layout_mirrors_the_reserve_step(tmp_path: Path) -> None:
    repo = tmp_path / "rulespec-us"
    repo.mkdir()
    runner_temp = tmp_path / "runner-temp"
    runner_temp.mkdir()

    layout = isolated_dependency_layout(PinnedWorkflow(PIN_92CCB0F2), repo, runner_temp)

    assert layout.isolated
    assert layout.stage is not None
    assert re.fullmatch(r"\.axiom-dependency-stage\.[A-Za-z0-9]{8}", layout.stage)
    assert (repo / layout.stage).is_dir()
    assert list((repo / layout.stage).iterdir()) == []
    assert layout.root.parent == runner_temp.resolve()
    assert re.fullmatch(r"axiom-dependencies\.[A-Za-z0-9]{8}", layout.root.name)
    assert stat.S_IMODE(layout.root.stat().st_mode) == 0o755
    assert not (repo / "_axiom").exists()
    assert layout.environment() == {
        "AXIOM_DEPENDENCY_STAGE": layout.stage,
        "AXIOM_DEPENDENCY_ROOT": str(layout.root),
    }


def test_isolated_layout_runs_the_steps_own_outside_check(tmp_path: Path) -> None:
    repo = tmp_path / "rulespec-us"
    repo.mkdir()

    # A runner temp inside the workspace puts the root inside the checkout.
    with pytest.raises(ValueError, match="must be outside the RuleSpec checkout"):
        isolated_dependency_layout(PinnedWorkflow(PIN_92CCB0F2), repo, repo)


def test_isolated_layout_never_reuses_a_directory(tmp_path: Path) -> None:
    repo = tmp_path / "rulespec-us"
    repo.mkdir()
    runner_temp = tmp_path / "runner-temp"
    runner_temp.mkdir()
    workflow = PinnedWorkflow(PIN_92CCB0F2)

    first = isolated_dependency_layout(workflow, repo, runner_temp)
    second = isolated_dependency_layout(workflow, repo, runner_temp)

    assert first.root != second.root
    assert first.stage != second.stage


def _dependency_paths(tmp_path: Path) -> dict[str, Path]:
    paths = {}
    for name in ("encode", "engine", "corpus", "rulespec_us", "compose"):
        path = tmp_path / "deps" / name
        path.mkdir(parents=True)
        paths[name] = path
    (paths["engine"] / "target" / "debug").mkdir(parents=True)
    binary = paths["engine"] / "target" / "debug" / "axiom-rules-engine"
    binary.write_text("#!/bin/sh\n")
    binary.chmod(0o755)
    return paths


@pytest.mark.parametrize("name", ["rulespec-us", "rulespec-us-tn"])
def test_isolated_links_follow_the_dependency_root(tmp_path: Path, name: str) -> None:
    parent = tmp_path / "checkout"
    parent.mkdir()
    repo = parent / name
    repo.mkdir()
    # A committed _axiom directory is plain content for this pin.
    (repo / "_axiom").mkdir()
    (repo / "_axiom" / "notes.txt").write_text("not a checkout\n")
    runner_temp = tmp_path / "runner-temp"
    runner_temp.mkdir()
    layout = isolated_dependency_layout(PinnedWorkflow(PIN_92CCB0F2), repo, runner_temp)
    paths = _dependency_paths(tmp_path)

    ci_parity._link_workflow_checkouts(repo, paths, layout)

    for key, checkout in {
        "encode": "axiom-encode",
        "engine": "axiom-rules-engine",
        "corpus": "axiom-corpus",
        "rulespec_us": "rulespec-us",
        "compose": "axiom-compose",
    }.items():
        assert (layout.root / checkout).resolve() == paths[key].resolve()
    siblings = {"axiom-rules-engine", "axiom-corpus"} | (
        set() if name == "rulespec-us" else {"rulespec-us"}
    )
    for sibling in siblings:
        link = parent / sibling
        assert link.is_symlink()
        assert Path(os.readlink(link)) == layout.root / sibling
    if name == "rulespec-us":
        assert not (parent / "rulespec-us").is_symlink()
    assert sorted(path.name for path in (repo / "_axiom").iterdir()) == ["notes.txt"]
    # What rulespec-us's own repository tests look for (test_az_na_basic_*).
    engine = (
        Path(layout.environment()["AXIOM_DEPENDENCY_ROOT"])
        / "axiom-rules-engine"
        / "target"
        / "debug"
        / "axiom-rules-engine"
    )
    assert engine.is_file()


def test_repository_pytest_sees_the_dependency_root(tmp_path: Path) -> None:
    repo = gates._repo(tmp_path, {"README.md": "x\n"}, name="rulespec-us")
    run = gates._run(repo, tmp_path, sha=PIN_92CCB0F2, plan=gates._plan("us"))
    runner_temp = tmp_path / "runner-temp-root"
    runner_temp.mkdir()
    run.layout = isolated_dependency_layout(run.workflow, repo, runner_temp)

    environment = ci_parity._repository_test_environment(
        run, interpreter=sys.executable
    )

    assert environment["AXIOM_DEPENDENCY_ROOT"] == str(run.layout.root)
    assert environment["AXIOM_DEPENDENCY_STAGE"] == run.layout.stage
    assert environment["PATH"].split(os.pathsep)[0] == str(
        run.layout.root / "axiom-rules-engine" / "target" / "debug"
    )


def test_workspace_pins_export_no_dependency_root(tmp_path: Path) -> None:
    repo = gates._repo(tmp_path, {"README.md": "x\n"}, name="rulespec-us")
    run = gates._run(repo, tmp_path, sha=PIN_6F11BE26, plan=gates._plan("us"))

    environment = ci_parity._repository_test_environment(
        run, interpreter=sys.executable
    )

    assert "AXIOM_DEPENDENCY_ROOT" not in environment
    assert "AXIOM_DEPENDENCY_STAGE" not in environment
    assert environment["PATH"].split(os.pathsep)[0] == str(
        run.paths["engine"] / "target" / "debug"
    )


def test_isolated_classifier_installs_from_the_dependency_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = gates._repo(tmp_path, {"README.md": "x\n"}, name="rulespec-us")
    run = gates._run(repo, tmp_path, sha=PIN_92CCB0F2, plan=gates._plan("us"))
    run.paths = _dependency_paths(tmp_path)
    (run.paths["encode"] / "pyproject.toml").write_text("[project]\nname='x'\n")
    runner_temp = tmp_path / "runner-temp-root"
    runner_temp.mkdir()
    run.layout = isolated_dependency_layout(run.workflow, repo, runner_temp)
    commands: list[list[str]] = []

    def fake_run(command, **_kwargs):
        commands.append(list(command))
        if command[:2] == ["uv", "venv"]:
            (Path(command[-1]) / "bin").mkdir(parents=True)
            (Path(command[-1]) / "bin" / "python").write_text("")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(ci_parity.subprocess, "run", fake_run)

    ci_parity._repository_test_python(run)

    classifier = run.layout.root / "axiom-encode-oracle-coverage"
    assert classifier.is_symlink()
    assert classifier.resolve() == run.paths["encode"].resolve()
    assert not (repo / "_axiom").exists()
    assert any(command[-1] == str(classifier) for command in commands)
    assert commands[0][:4] == ["uv", "venv", "--python", "3.14+gil"]


# ---------------------------------------------------------------------------
# Composer arguments
# ---------------------------------------------------------------------------


@settings(max_examples=60, deadline=None)
@given(
    composer=st.booleans(),
    name=st.sampled_from(["rulespec-us", "rulespec-us-tn", "rulespec-uk", "x"]),
    isolated=st.booleans(),
)
def test_composer_arguments_follow_the_workflows_bash_conditions(
    composer: bool, name: str, isolated: bool
) -> None:
    run = ci_parity.WorkflowRun.__new__(ci_parity.WorkflowRun)
    run.repo = Path("/work") / name
    run.paths = {"rulespec_us": Path("/deps/rulespec-us")}
    run.composer = Path("/runtime/axiom-compose") if composer else None
    run.layout = (
        DependencyLayout(Path("/runner/axiom-dependencies.abcdefgh"), ".stage")
        if isolated
        else None
    )

    arguments = ci_parity._composer_arguments(run)

    # composer_args: [ -n "$AXIOM_COMPOSE_REF" ]; dependency_args: also
    # [ "$(basename "$GITHUB_WORKSPACE")" != "rulespec-us" ].
    expected: tuple[str, ...] = ()
    if composer:
        expected = ("--axiom-compose-path", "/runtime/axiom-compose")
        if name != "rulespec-us":
            # Always the real checkout: the encoder refuses a symlinked root.
            expected += ("--rulespec-dependency-root", "/deps/rulespec-us")
    assert arguments == expected


def _composer_run(tmp_path: Path, name: str = "rulespec-us") -> Any:
    repo = gates._repo(
        tmp_path,
        {"us/statutes/a.yaml": gates.PLAIN, "us/statutes/a.test.yaml": "cases: []\n"},
        name=name,
    )
    run = gates._run(repo, tmp_path, sha=PIN_92CCB0F2, plan=gates._plan("us"))
    run.composer = tmp_path / "runtime" / "axiom-compose"
    run.selections = {
        "us": ci_parity.ShardSelection(
            "us", ("us",), ("us/statutes/a.yaml",), ("us/statutes/a.test.yaml",)
        )
    }
    return run


@pytest.mark.parametrize("name", ["rulespec-us", "rulespec-us-tn"])
def test_validate_and_companion_tests_pass_the_composer_where_ci_does(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    run = _composer_run(tmp_path, name)
    recorder = gates._BatchRecorder()
    monkeypatch.setattr(ci_parity, "_run_cli_batch", recorder)

    ci_parity._gate_validate(run, gates._spec("validate", PIN_92CCB0F2))
    ci_parity._gate_companion_tests(run, gates._spec("companion_tests", PIN_92CCB0F2))

    validate, test = (batch[0].arguments for batch in recorder.batches)
    composer = ["--axiom-compose-path", str(run.composer)]
    if name != "rulespec-us":
        composer += ["--rulespec-dependency-root", str(run.paths["rulespec_us"])]
    assert list(validate) == [
        "validate",
        str(run.repo / "us/statutes/a.yaml"),
        "--skip-reviewers",
        *composer,
        "--corpus-path",
        str(run.paths["corpus"]),
        "--axiom-rules-engine-path",
        str(run.paths["engine"]),
    ]
    assert list(test) == [
        "test",
        *composer,
        "--root",
        str(run.repo / "us"),
        "--axiom-rules-engine-path",
        str(run.paths["engine"]),
        "statutes/a.test.yaml",
    ]


def test_without_a_composer_the_invocations_are_6f11be26s(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = _composer_run(tmp_path)
    run.composer = None
    recorder = gates._BatchRecorder()
    monkeypatch.setattr(ci_parity, "_run_cli_batch", recorder)

    ci_parity._gate_validate(run, gates._spec("validate", PIN_92CCB0F2))

    assert "--axiom-compose-path" not in recorder.batches[0][0].arguments


# ---------------------------------------------------------------------------
# The composer runtime build
# ---------------------------------------------------------------------------


def _build_step_commands() -> list[list[str]]:
    """The uv command lines of "Build protected axiom-compose runtime"."""

    run = PinnedWorkflow(PIN_92CCB0F2).step(
        "validate", "Build protected axiom-compose runtime"
    )["run"]
    logical = run.replace("\\\n", " ")
    return [
        line.split() for line in logical.splitlines() if line.strip().startswith("uv ")
    ]


def test_composer_build_runs_the_steps_uv_commands_in_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    encode = gates._repo(
        tmp_path,
        {
            "scripts/provision_axiom_compose_runtime.py": (
                "import sys, pathlib\n"
                "args = dict(zip(sys.argv[1::2], sys.argv[2::2]))\n"
                "destination = pathlib.Path(args['--destination'])\n"
                "destination.mkdir()\n"
                "launcher = destination / 'axiom-compose'\n"
                "launcher.write_text('#!/bin/sh\\nexit 0\\n')\n"
                "launcher.chmod(0o777)\n"
            )
        },
        name="axiom-encode",
    )
    pin = gates._git(encode, "rev-parse", "HEAD")
    calls: list[tuple[list[str], dict[str, str]]] = []
    cwds: list[tuple[list[str], Path]] = []
    real_run = ci_parity._run_build_command

    def fake(command, *, cwd, environment):
        calls.append((list(command), dict(environment)))
        cwds.append((list(command), cwd))
        if command[0] == "uv":
            if command[1] == "venv":
                # The build venv: its python is this test's interpreter.
                (Path(command[-1]) / "bin").mkdir(parents=True)
                (Path(command[-1]) / "bin" / "python").symlink_to(sys.executable)
            return subprocess.CompletedProcess(command, 0, "uv 0.11.7\n", None)
        return real_run(command, cwd=cwd, environment=environment)

    monkeypatch.setattr(ci_parity, "_run_build_command", fake)
    monkeypatch.setenv("UV_INDEX_URL", "https://ambient.example/simple")
    compose = tmp_path / "compose"
    compose.mkdir()
    temp = tmp_path / "temp"

    workspace = tmp_path / "rulespec-us"
    workspace.mkdir()
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "developer-config"))

    launcher = ci_parity._build_workflow_composer(
        compose,
        COMPOSE_REF,
        encode,
        pin,
        temp,
        workspace=workspace,
        offline=False,
        notes=[],
    )

    assert launcher == temp / "axiom-compose-verification" / "axiom-compose"
    uv_calls = [command for command, _ in calls if command[0] == "uv"]
    workflow_uv = _build_step_commands()
    # Same subcommands, in the same order, with the same literal options.
    assert [command[:3] for command in uv_calls[:4]] == [
        command[:3] for command in workflow_uv
    ]
    assert uv_calls[1][:4] == ["uv", "venv", "--python", "3.14+gil"]
    for local, workflow in zip(uv_calls, workflow_uv, strict=False):
        literal = {token for token in workflow if token.startswith("--")}
        assert literal <= set(local), (local, workflow)
        # CI's uv reads committed project configuration, so it is not off.
        assert "--no-config" not in local
    empty = temp / "axiom-compose-empty-config"
    for command, environment in calls:
        if command[0] == "uv":
            assert not any(name.startswith("UV_") for name in environment)
            # A fresh runner has no user or system uv configuration.
            assert environment["XDG_CONFIG_HOME"] == str(empty)
            assert environment["XDG_CONFIG_DIRS"] == str(empty)
    assert list(empty.iterdir()) == []
    # The step runs in the workspace, where uv discovers project config.
    for command, cwd in cwds[:4]:
        assert cwd == workspace, command
    provision = next(
        (command, environment)
        for command, environment in calls
        if command[1].endswith("provision_axiom_compose_runtime.py")
    )
    assert provision[0][2:] == [
        "--destination",
        str(temp / "axiom-compose-verification"),
        "--site-packages",
        str(temp / "axiom-compose-site-packages"),
        "--compose-checkout",
        str(compose),
        "--compose-ref",
        COMPOSE_REF,
    ]
    assert provision[1]["PATH"] == "/usr/bin:/bin:/usr/sbin:/sbin"
    # The pin's committed script ran, not a worktree copy.
    assert provision[0][1] == str(
        temp
        / "axiom-compose-provision"
        / "scripts"
        / "provision_axiom_compose_runtime.py"
    )
    # sudo chmod -R go-w
    assert stat.S_IMODE(launcher.stat().st_mode) & (stat.S_IWGRP | stat.S_IWOTH) == 0


def test_composer_build_fails_closed_when_the_encoder_pin_cannot_provision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    encode = gates._repo(tmp_path, {"README.md": "no scripts\n"}, name="axiom-encode")
    pin = gates._git(encode, "rev-parse", "HEAD")
    monkeypatch.setattr(
        ci_parity,
        "_run_build_command",
        lambda command, *, cwd, environment: subprocess.CompletedProcess(
            command, 0, "", None
        ),
    )

    with pytest.raises(ValueError, match="composer-aware encoder|has no scripts/"):
        ci_parity._build_workflow_composer(
            tmp_path / "compose",
            COMPOSE_REF,
            encode,
            pin,
            tmp_path / "temp",
            workspace=tmp_path,
            offline=False,
        )


def test_composer_build_reports_a_failing_uv_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        ci_parity,
        "_run_build_command",
        lambda command, *, cwd, environment: subprocess.CompletedProcess(
            command, 2 if command[1] == "export" else 0, "lock is out of date\n", None
        ),
    )

    with pytest.raises(ValueError, match="uv export .* lock is out of date"):
        ci_parity._build_workflow_composer(
            tmp_path / "compose",
            COMPOSE_REF,
            tmp_path / "encode",
            "0" * 40,
            tmp_path / "temp",
            workspace=tmp_path,
            offline=False,
        )


def _tiny_compose_repository(tmp_path: Path) -> tuple[Path, str]:
    repo = gates._repo(
        tmp_path,
        {
            "pyproject.toml": (
                "[project]\n"
                'name = "axiom-compose"\n'
                'version = "0.0.1"\n'
                'requires-python = ">=3.14"\n'
                "dependencies = []\n"
            ),
            "src/axiom_compose/__init__.py": "",
            "src/axiom_compose/cli.py": (
                "def main() -> int:\n    print('axiom-compose ok')\n    return 0\n"
            ),
        },
        name="axiom-compose-source",
    )
    subprocess.run(
        ["uv", "lock", "--no-config", "--offline", "--project", str(repo)],
        check=True,
        capture_output=True,
    )
    return repo, gates._commit(repo, message="lock")


# Opt-in: like CI's step, it runs `uv python install 3.14` against the
# developer's uv installation and copies a whole Python runtime.
@pytest.mark.skipif(
    os.environ.get("AXIOM_CI_PARITY_REAL_COMPOSER") != "1",
    reason="set AXIOM_CI_PARITY_REAL_COMPOSER=1 to build a real composer runtime",
)
@pytest.mark.skipif(shutil.which("uv") is None, reason="needs uv")
@pytest.mark.skipif(
    sys.platform == "linux" and shutil.which("patchelf") is None,
    reason="the provision script needs patchelf on Linux",
)
def test_composer_runtime_builds_with_the_real_provision_script(tmp_path: Path) -> None:
    source, ref = _tiny_compose_repository(tmp_path)
    clone = tmp_path / "clone" / "axiom-compose"
    clone.parent.mkdir()
    subprocess.run(
        ["git", "clone", "-q", "--no-local", str(source), str(clone)], check=True
    )
    gates._git(clone, "checkout", "-q", "--detach", ref)
    gates._git(
        clone,
        "remote",
        "set-url",
        "origin",
        "https://github.com/" + ci_parity.COMPOSE_REPOSITORY,
    )
    encode = gates._repo(
        tmp_path,
        {
            f"scripts/{name}": (REPOSITORY_ROOT / "scripts" / name).read_text()
            for name in (
                "provision_axiom_compose_runtime.py",
                "provision_verification_supervisor.py",
            )
        },
        name="axiom-encode-pin",
    )
    pin = gates._git(encode, "rev-parse", "HEAD")
    notes: list[str] = []

    try:
        launcher = ci_parity._build_workflow_composer(
            clone,
            ref,
            encode,
            pin,
            tmp_path / "temp",
            workspace=clone.parent,
            offline=False,
            notes=notes,
        )
    except ValueError as exc:
        if "uv python install" in str(exc) or "uv venv" in str(exc):
            pytest.skip(f"no managed Python 3.14 available: {exc}")
        raise

    result = subprocess.run([str(launcher)], capture_output=True, text=True, check=True)
    assert result.stdout == "axiom-compose ok\n"
    attestation = json.loads(
        (launcher.parent / "runtime-attestation.json").read_text(encoding="utf-8")
    )
    assert attestation["commit"] == ref
    assert notes and notes[0].startswith(f"composer runtime: axiom-compose {ref}")


# ---------------------------------------------------------------------------
# run_ci wiring
# ---------------------------------------------------------------------------


def _harness(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    sha: str = PIN_92CCB0F2,
    compose: str = COMPOSE_REF,
) -> tuple[Any, list[tuple[Any, ...]], Path]:
    from tests.test_ci_parity_runtime import _CiHarness

    harness = _CiHarness(tmp_path, monkeypatch, workflow_sha=sha)
    if "axiom-compose-ref" in SUPPORTED_WORKFLOW_PINS[sha].inputs:
        harness.caller = replace(
            harness.caller,
            refs={**harness.caller.refs, **({"compose": compose} if compose else {})},
            inputs={**harness.caller.inputs, "axiom-compose-ref": compose},
        )
    events: list[tuple[Any, ...]] = []
    # The local uv is checked against the workflow's pin in its own tests.
    monkeypatch.setattr(ci_parity, "composer_uv_mismatch", lambda workflow: None)
    monkeypatch.setattr(ci_parity, "composer_uv_config_mismatch", lambda: None)
    monkeypatch.setattr(
        ci_parity,
        "validate_composer_pin",
        lambda workflow, repo, ref: events.append(("validate_composer_pin", ref)),
    )

    def build(checkout, ref, encode, pin, temp, *, workspace, offline, notes=None):
        events.append(("build_composer", checkout, ref, encode, pin, offline))
        assert workspace == harness.repo
        return temp / "axiom-compose-verification" / "axiom-compose"

    monkeypatch.setattr(ci_parity, "_build_workflow_composer", build)
    compose_checkout = tmp_path / "axiom-compose"
    compose_checkout.mkdir()
    return harness, events, compose_checkout


def test_run_ci_builds_the_composer_and_isolates_dependencies(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    harness, events, compose = _harness(tmp_path, monkeypatch)

    code = run_ci(harness.args("--compose-path", str(compose)))
    report = json.loads(capsys.readouterr().out)

    assert code == 0, report
    assert [event[0] for event in events] == [
        "validate_composer_pin",
        "build_composer",
    ]
    assert events[0] == ("validate_composer_pin", COMPOSE_REF)
    _, checkout, ref, encode, pin, offline = events[1]
    assert (ref, pin, offline) == (COMPOSE_REF, harness.caller.refs["encode"], False)
    assert encode == harness.paths["encode"].resolve()
    verified = [call for call in harness.calls["verify_dependency_checkout"]]
    assert [call[0] for call in verified][-1] == "compose"
    assert verified[-1][1:3] == (compose.resolve(), COMPOSE_REF)
    # The engine builds outside the rules checkout, as CI's does.
    assert harness.rules_checkout is None
    run = harness.run
    assert run is not None
    assert run.layout is not None and run.layout.isolated
    assert harness.layout is run.layout
    assert (harness.repo / run.layout.stage).is_dir()
    # The composer and canonical rulespec-us entries are fresh checkouts moved
    # into the root (the harness's checkout stand-in yields the source).
    assert checkout == run.layout.root / "axiom-compose"
    assert run.paths["compose"] == run.layout.root / "axiom-compose"
    assert run.paths["rulespec_us"] == run.layout.root / "rulespec-us"
    assert run.runner_temp == run.layout.root.parent
    # The bound engine view is the dependency-root entry itself, not a link.
    assert run.paths["engine"] == run.layout.root / "axiom-rules-engine"
    assert harness.calls["_link_workflow_checkouts"][0][1]["engine"] == (
        run.layout.root / "axiom-rules-engine"
    )
    assert run.composer is not None and run.composer.name == "axiom-compose"
    assert not (harness.repo / "_axiom").exists()


def test_run_ci_without_a_composer_ref_skips_every_composer_step(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    harness, events, _ = _harness(tmp_path, monkeypatch, compose="")

    assert run_ci(harness.args()) == 0, capsys.readouterr().out

    assert events == []
    assert "compose" not in [
        call[0] for call in harness.calls["verify_dependency_checkout"]
    ]
    assert harness.run is not None
    assert harness.run.composer is None
    assert harness.run.layout is not None and harness.run.layout.isolated


def test_run_ci_reports_a_failing_composer_pin_as_resolution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    harness, _, compose = _harness(tmp_path, monkeypatch)

    def refuse(workflow, repo, ref):
        raise ValueError("Validate composer pin failed: composer input must match")

    monkeypatch.setattr(ci_parity, "validate_composer_pin", refuse)

    assert run_ci(harness.args("--compose-path", str(compose))) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["verdict"] == "FAIL"
    assert "composer input must match" in report["resolution_error"]
    # The shards job completed and stays in the report.
    assert [gate["gate"] for gate in report["gates"]][-1] == "compute_shards"


def test_workspace_pins_keep_the_legacy_layout_and_engine_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    harness, events, _ = _harness(tmp_path, monkeypatch, sha=PIN_6F11BE26)

    assert run_ci(harness.args()) == 0, capsys.readouterr().out

    assert events == []
    assert harness.rules_checkout == harness.repo
    assert harness.layout is None
    assert harness.run is not None and harness.run.layout is None


def test_compose_path_is_a_ci_flag(tmp_path: Path) -> None:
    from tests.test_ci_parity_runtime import _ci_parser

    args = _ci_parser().parse_args(
        [
            "ci",
            "--repo",
            str(tmp_path),
            "--corpus-release-public-key",
            "k",
            "--compose-path",
            str(tmp_path / "compose"),
        ]
    )

    assert args.compose_path == tmp_path / "compose"
    assert ci_parity.resolve_compose_path(args, tmp_path / "rulespec-us") == (
        (tmp_path / "compose").resolve()
    )
    args.compose_path = None
    assert ci_parity.resolve_compose_path(args, tmp_path / "rulespec-us") == (
        (tmp_path / "axiom-compose").resolve()
    )


@settings(max_examples=25, deadline=None)
@given(
    prefix=st.sampled_from([".axiom-dependency-stage.", "axiom-dependencies."]),
    count=st.integers(min_value=1, max_value=12),
)
def test_exclusive_directories_are_new_mktemp_names(
    tmp_path_factory: pytest.TempPathFactory, prefix: str, count: int
) -> None:
    parent = tmp_path_factory.mktemp("exclusive")
    # A pre-existing directory of the same shape is never reused.
    squatter = parent / f"{prefix}AAAAAAAA"
    squatter.mkdir()

    made = [ci_parity._exclusive_directory(parent, prefix) for _ in range(count)]

    assert len(set(made)) == count
    assert squatter not in made
    for path in made:
        assert path.parent == parent
        assert re.fullmatch(re.escape(prefix) + r"[A-Za-z0-9]{8}", path.name)
        assert path.is_dir() and list(path.iterdir()) == []
        assert stat.S_IMODE(path.stat().st_mode) == 0o700


def test_dependency_root_argument_is_the_fresh_entry(tmp_path: Path) -> None:
    import contextlib

    source, commit = _canonical_rulespec_us(tmp_path)
    run = _composer_run(tmp_path, "rulespec-us-tn")
    runner_temp = (tmp_path / "runner-temp-root").resolve()
    runner_temp.mkdir()
    run.layout = isolated_dependency_layout(run.workflow, run.repo, runner_temp)

    with contextlib.ExitStack() as resources:
        run.paths["rulespec_us"] = ci_parity._fresh_dependency_entry(
            resources,
            run.layout,
            "rulespec_us",
            source,
            commit,
            "TheAxiomFoundation/rulespec-us",
        )
        arguments = ci_parity._composer_arguments(run)

        root = Path(arguments[arguments.index("--rulespec-dependency-root") + 1])
        assert root == run.layout.root / "rulespec-us"
        # No component of the path is a link, as the encoder's walk requires.
        cursor = Path(root.anchor)
        for part in root.parts[1:]:
            cursor /= part
            assert not cursor.is_symlink(), cursor


def test_a_mismatched_composer_checkout_builds_its_own_head(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    harness, events, compose = _harness(tmp_path, monkeypatch)
    harness.mismatches["compose"] = ci_parity.DependencyMismatch(
        "compose", "e" * 40, COMPOSE_REF
    )

    code = run_ci(harness.args("--compose-path", str(compose), "--allow-ref-mismatch"))
    report = json.loads(capsys.readouterr().out)

    # The harness resolves refs to themselves, so HEAD stays "HEAD".
    assert code == 3, report
    assert events[1][2] == "HEAD"
    assert report["verdict"] == "PASS-WITH-MISMATCHED-DEPS"
    assert [item["name"] for item in report["dependency_mismatches"]] == ["compose"]


# ---------------------------------------------------------------------------
# Cycle-1 review fixes
# ---------------------------------------------------------------------------


def _canonical_rulespec_us(tmp_path: Path) -> tuple[Path, str]:
    repo = gates._repo(
        tmp_path,
        {"us/statutes/x.yaml": gates.PLAIN, "README.md": "canonical\n"},
        name="rulespec-us-source",
        remote="https://github.com/TheAxiomFoundation/rulespec-us.git",
    )
    return repo, gates._git(repo, "rev-parse", "HEAD")


def test_fresh_entries_are_clean_checkouts_the_encoder_accepts(tmp_path: Path) -> None:
    import contextlib

    from axiom_encode.harness.validator_pipeline import (
        _normalize_rulespec_dependency_roots,
    )

    source, commit = _canonical_rulespec_us(tmp_path)
    # Local state CI's fresh checkout never has: an untracked symlink.
    (source / ".venv").mkdir()
    (source / ".venv" / "python").symlink_to(sys.executable)
    (source / "scratch.yaml").write_text("untracked\n")
    runner_temp = tmp_path / "runner-temp-root"
    runner_temp.mkdir()
    repo = tmp_path / "checkout" / "rulespec-us-tn"
    repo.mkdir(parents=True)
    layout = isolated_dependency_layout(PinnedWorkflow(PIN_92CCB0F2), repo, runner_temp)

    with contextlib.ExitStack() as resources:
        entry = ci_parity._fresh_dependency_entry(
            resources,
            layout,
            "rulespec_us",
            source,
            commit,
            "TheAxiomFoundation/rulespec-us",
        )

        assert entry == layout.root / "rulespec-us"
        assert entry.is_dir() and not entry.is_symlink()
        assert gates._git(entry, "rev-parse", "HEAD") == commit
        assert gates._git(entry, "remote", "get-url", "origin") == (
            "https://github.com/TheAxiomFoundation/rulespec-us"
        )
        assert not (entry / ".venv").exists()
        assert not (entry / "scratch.yaml").exists()
        assert _normalize_rulespec_dependency_roots([entry]) == (entry.resolve(),)
        # The local checkout itself, with its untracked symlink, is refused,
        # and so is a link to it: both are what ci passed before.
        with pytest.raises(Exception, match="symlink"):
            _normalize_rulespec_dependency_roots([source])
        link = tmp_path / "linked" / "rulespec-us"
        link.parent.mkdir()
        link.symlink_to(entry)
        with pytest.raises(Exception, match="symlink"):
            _normalize_rulespec_dependency_roots([link])


def test_fresh_entry_never_replaces_an_existing_path(tmp_path: Path) -> None:
    import contextlib

    source, commit = _canonical_rulespec_us(tmp_path)
    runner_temp = tmp_path / "runner-temp-root"
    runner_temp.mkdir()
    repo = tmp_path / "rulespec-us"
    repo.mkdir()
    layout = isolated_dependency_layout(PinnedWorkflow(PIN_92CCB0F2), repo, runner_temp)
    (layout.root / "rulespec-us").mkdir()

    with (
        contextlib.ExitStack() as resources,
        pytest.raises(ValueError, match="already exists"),
    ):
        ci_parity._fresh_dependency_entry(
            resources,
            layout,
            "rulespec_us",
            source,
            commit,
            "TheAxiomFoundation/rulespec-us",
        )


def test_pinned_scripts_ignore_local_attributes_and_worktree(tmp_path: Path) -> None:
    encode = gates._repo(
        tmp_path,
        {
            "scripts/provision_axiom_compose_runtime.py": "print('pinned')\n",
            "scripts/helper.py": "VALUE = 1\n",
            "src/other.py": "x = 1\n",
        },
        name="axiom-encode",
    )
    pin = gates._git(encode, "rev-parse", "HEAD")
    # Local state a CI checkout of the pin never has.
    (encode / ".git" / "info").mkdir(exist_ok=True)
    (encode / ".git" / "info" / "attributes").write_text("scripts/** export-ignore\n")
    (encode / "scripts" / "helper.py").write_text("VALUE = 'edited'\n")
    destination = tmp_path / "scripts-out"

    ci_parity._write_pinned_tree(encode, pin, "scripts", destination)

    assert sorted(
        path.relative_to(destination).as_posix()
        for path in destination.rglob("*")
        if path.is_file()
    ) == ["scripts/helper.py", "scripts/provision_axiom_compose_runtime.py"]
    assert (destination / "scripts" / "helper.py").read_text() == "VALUE = 1\n"


@pytest.mark.parametrize(
    ("output", "code", "expected"),
    [
        ("uv 0.11.7 (9d177269e 2026-04-15 aarch64-apple-darwin)\n", 0, None),
        ("uv 0.11.8 (abc 2026-05-01)\n", 0, ("uv 0.11.8", "uv 0.11.7")),
        ("", 127, ("uv unavailable", "uv 0.11.7")),
    ],
    ids=["pinned", "newer", "missing"],
)
def test_composer_uv_is_the_version_setup_uv_installs(
    monkeypatch: pytest.MonkeyPatch,
    output: str,
    code: int,
    expected: tuple[str, str] | None,
) -> None:
    seen: list[dict[str, str]] = []

    def fake_run(command, **kwargs):
        assert command == ["uv", "--version"]
        seen.append(kwargs["env"])
        return subprocess.CompletedProcess(command, code, output, None)

    monkeypatch.setattr(ci_parity.subprocess, "run", fake_run)
    monkeypatch.setenv("UV_PYTHON", "3.12")

    mismatch = ci_parity.composer_uv_mismatch(PinnedWorkflow(PIN_92CCB0F2))

    assert (
        None if mismatch is None else (mismatch.head_sha, mismatch.pinned_sha)
    ) == expected
    assert "UV_PYTHON" not in seen[0]
    assert ci_parity.composer_uv_mismatch(PinnedWorkflow(PIN_6F11BE26)) is None


def test_run_ci_refuses_another_uv_without_the_development_flag(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    harness, events, compose = _harness(tmp_path, monkeypatch)
    monkeypatch.setattr(
        ci_parity,
        "composer_uv_mismatch",
        lambda workflow: ci_parity.DependencyMismatch(
            "composer-uv", "uv 0.12.0", "uv 0.11.7"
        ),
    )

    assert run_ci(harness.args("--compose-path", str(compose))) == 1
    report = json.loads(capsys.readouterr().out)
    assert "UV MISMATCH: composer-uv" in report["resolution_error"]
    assert [event[0] for event in events] == ["validate_composer_pin"]


def test_executor_uses_the_runs_runner_temp(tmp_path: Path) -> None:
    repo = gates._repo(tmp_path, {"README.md": "x\n"}, name="rulespec-us")
    run = gates._run(repo, tmp_path, sha=PIN_92CCB0F2, plan=gates._plan("us"))
    runner_temp = tmp_path / "shared-runner-temp"
    runner_temp.mkdir()
    run.runner_temp = runner_temp
    seen: list[Path] = []

    def first(run_, spec):
        seen.append(run_.temp)
        return ci_parity.GateResult(spec.key, spec.name, "FAIL")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(ci_parity, "_gate_unsupported_paths", first)
        ci_parity.execute_workflow_gates(run)

    assert seen == [runner_temp]
    # The caller owns its lifetime, so the executor leaves it in place.
    assert runner_temp.is_dir()


# ---------------------------------------------------------------------------
# Cycle-2 review fixes
# ---------------------------------------------------------------------------


def test_a_system_uv_configuration_is_a_composer_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tmp_path / "etc" / "uv" / "uv.toml"
    monkeypatch.setattr(ci_parity, "SYSTEM_UV_CONFIG", config)
    assert ci_parity.composer_uv_config_mismatch() is None

    config.parent.mkdir(parents=True)
    config.write_text('exclude-newer = "2024-01-01"\n')
    mismatch = ci_parity.composer_uv_config_mismatch()

    assert mismatch == ci_parity.DependencyMismatch(
        "composer-uv-config", f"system uv configuration {config}", "none"
    )
    # A dangling link is still configuration uv would try to read.
    config.unlink()
    config.symlink_to(tmp_path / "missing.toml")
    assert ci_parity.composer_uv_config_mismatch() is not None


@pytest.mark.parametrize("allow", [False, True])
def test_run_ci_refuses_a_system_uv_configuration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    allow: bool,
) -> None:
    harness, events, compose = _harness(tmp_path, monkeypatch)
    monkeypatch.setattr(
        ci_parity,
        "composer_uv_config_mismatch",
        lambda: ci_parity.DependencyMismatch(
            "composer-uv-config", "system uv configuration /etc/uv/uv.toml", "none"
        ),
    )

    if not allow:
        assert run_ci(harness.args("--compose-path", str(compose))) == 1
        assert (
            "UV MISMATCH: composer-uv-config"
            in json.loads(capsys.readouterr().out)["resolution_error"]
        )
        assert [event[0] for event in events] == ["validate_composer_pin"]
        return
    assert (
        run_ci(harness.args("--compose-path", str(compose), "--allow-encoder-mismatch"))
        == 3
    )
    report = json.loads(capsys.readouterr().out)
    assert "composer-uv-config" in [
        item["name"] for item in report["dependency_mismatches"]
    ]


def test_pinned_scripts_ignore_local_replace_refs(tmp_path: Path) -> None:
    encode = gates._repo(
        tmp_path,
        {"scripts/provision_axiom_compose_runtime.py": "print('pinned')\n"},
        name="axiom-encode",
    )
    pin = gates._git(encode, "rev-parse", "HEAD")
    pinned_blob = gates._git(
        encode, "rev-parse", f"{pin}:scripts/provision_axiom_compose_runtime.py"
    )
    edited = (
        subprocess.run(
            ["git", "-C", str(encode), "hash-object", "-w", "--stdin"],
            input=b"print('replaced')\n",
            stdout=subprocess.PIPE,
            check=True,
        )
        .stdout.decode()
        .strip()
    )
    gates._git(encode, "replace", pinned_blob, edited)
    destination = tmp_path / "scripts-out"

    ci_parity._write_pinned_tree(encode, pin, "scripts", destination)

    assert (
        destination / "scripts" / "provision_axiom_compose_runtime.py"
    ).read_text() == "print('pinned')\n"


def test_pinned_scripts_refuse_entries_ci_would_materialize_differently(
    tmp_path: Path,
) -> None:
    encode = gates._repo(
        tmp_path,
        {"scripts/provision_axiom_compose_runtime.py": "print('pinned')\n"},
        name="axiom-encode",
    )
    (encode / "scripts" / "helper.py").symlink_to("provision_axiom_compose_runtime.py")
    pin = gates._commit(encode, message="symlinked helper")

    with pytest.raises(ValueError, match="non-regular scripts/ entry .* 120000"):
        ci_parity._write_pinned_tree(encode, pin, "scripts", tmp_path / "out")
