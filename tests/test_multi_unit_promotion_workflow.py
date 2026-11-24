"""The signing workflow packages every unit of a multi-unit promotion (#2301).

The promotion command writes one signed manifest per unit. These tests run
the workflow's own embedded packaging code and shell fragments and prove that
all ``k`` manifests are expected, checked and recorded, that a manifest of an
unexpected citation fails the run, and that a single-unit run is unchanged.
"""

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from axiom_encode.cli import APPLIED_ENCODING_MANIFEST_SCHEMA

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github/workflows/targeted-signed-reencode.yml"
TOOL = "axiom-encode promote-reviewed-candidate"
NAMES = ("a", "b", "c")
# Turns the multi-unit switch on inside a child interpreter. The switch has no
# environment variable; a test can only replace the constant.
SWITCH_ON = (
    "from axiom_encode import reviewed_pr_admission\n"
    "reviewed_pr_admission.MULTI_UNIT_PROMOTION_ENABLED = True\n"
)


def citation(name: str) -> str:
    return f"us/statute/7/2015/{name}"


def primary(name: str) -> str:
    return f"us/statutes/7/2015/{name}.yaml"


def manifest(name: str) -> str:
    return f".axiom/encoding-manifests/us/statutes/7/2015/{name}.json"


def step(name: str) -> dict:
    workflow = yaml.safe_load(WORKFLOW.read_text())
    return next(
        item for item in workflow["jobs"]["encode"]["steps"] if item.get("name") == name
    )


def package_script() -> str:
    marker = (
        '"${workflow_python[@]}" - \\\n'
        '  "$artifact/context-manifest.json" \\\n'
        "  \"$artifact/apply-manifests.json\" <<'PY'\n"
    )
    return (
        step("Package exact generated changes")["run"]
        .split(marker, 1)[1]
        .split('\nPY\n"${workflow_python[@]}" - "$artifact/metadata.json"', 1)[0]
    )


def metadata_script() -> str:
    marker = '"${workflow_python[@]}" - "$artifact/metadata.json" <<\'PY\'\n'
    return (
        step("Package exact generated changes")["run"]
        .split(marker, 1)[1]
        .split('\nPY\ntest -s "$artifact/status.txt"', 1)[0]
    )


def wrapper(units=None) -> str:
    payload = {
        "schema": "axiom-encode/reviewed-pr-promotion/v1",
        "rulespec_pr": 42,
        "transaction": {
            "schema": "axiom-encode/atomic-source-transaction/v5",
            "source_bundle": [],
            "canonical_refresh_bundle": [],
            "primary_required_test_cases": [],
            "require_complete_source_unit": True,
            "manifest_only_refresh": False,
            "reviewed_candidate_promotion": True,
        },
    }
    if units is not None:
        payload["units"] = units
    return json.dumps(payload)


def init(repository: Path) -> str:
    repository.mkdir(parents=True)
    for arguments in (
        ["init", "-q"],
        ["config", "user.email", "test@example.com"],
        ["config", "user.name", "Test"],
    ):
        subprocess.run(["git", *arguments], cwd=repository, check=True)
    (repository / "README.md").write_text("fixture\n")
    subprocess.run(["git", "add", "README.md"], cwd=repository, check=True)
    subprocess.run(["git", "commit", "-qm", "fixture"], cwd=repository, check=True)
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repository, text=True
    ).strip()


def child_env(tmp_path: Path, *, switch_on: bool, **extra: str) -> dict:
    """The environment of a child interpreter that runs workflow code."""
    site = tmp_path / ("site-on" if switch_on else "site-off")
    site.mkdir(exist_ok=True)
    if switch_on:
        (site / "sitecustomize.py").write_text(SWITCH_ON)
    return {
        **os.environ,
        "PYTHONPATH": os.pathsep.join([str(site), str(ROOT / "src")]),
        **extra,
    }


class Packaging:
    """One run of the workflow's packaging code over a promoted checkout."""

    def __init__(self, tmp_path: Path, names=NAMES):
        self.tmp_path = tmp_path
        self.rulespec = tmp_path / "rulespec-us"
        self.ref = init(self.rulespec)
        (tmp_path / "source-bundle.json").write_text("[]\n")
        (tmp_path / "canonical-refresh-bundle.json").write_text("[]\n")
        self.payloads = {}
        for name in names:
            self.write(name)
        self.artifact = tmp_path / "artifact"
        self.artifact.mkdir()

    def write(self, name: str, *, at: str | None = None, **changes) -> None:
        payload = {
            "schema_version": APPLIED_ENCODING_MANIFEST_SCHEMA,
            "tool": TOOL,
            "citation": citation(name),
            "reviewed_rulespec_ref": self.ref,
            "applied_files": [
                {"path": primary(name), "sha256": "1" * 64},
                {
                    "path": primary(name).removesuffix(".yaml") + ".test.yaml",
                    "sha256": "2" * 64,
                },
            ],
            **changes,
        }
        path = self.rulespec / manifest(at or name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        self.payloads[at or name] = payload

    def run(self, units, *, lead="a", switch_on=True, script=None):
        event = self.tmp_path / "event.json"
        event.write_text(
            json.dumps({"inputs": {"citation": citation(lead), "review_finding": ""}})
        )
        return subprocess.run(
            [
                sys.executable,
                "-",
                str(self.artifact / "context-manifest.json"),
                str(self.artifact / "apply-manifests.json"),
            ],
            cwd=self.tmp_path,
            input=script or package_script(),
            check=False,
            capture_output=True,
            text=True,
            env=child_env(
                self.tmp_path,
                switch_on=switch_on,
                CITATION=citation(lead),
                REVIEW_FINDING="",
                RUNNER_TEMP=str(self.tmp_path),
                RULESPEC_CHECKOUT="rulespec-us",
                RULESPEC_REF=self.ref,
                REPLACE_RULESPEC_PATH=primary(lead),
                ATOMIC_SOURCE_JSON=wrapper(units),
                GITHUB_EVENT_PATH=str(event),
                GITHUB_SHA="a" * 40,
            ),
        )

    def inventory(self) -> list:
        payload = json.loads((self.artifact / "apply-manifests.json").read_text())
        assert payload["schema"] == "axiom-encode/applied-manifest-inventory/v1"
        return payload["items"]

    def item(self, name: str) -> dict:
        path = self.rulespec / manifest(name)
        return {
            "citation": citation(name),
            "path": manifest(name),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }

    def evidence(self, file_name: str) -> dict:
        return json.loads((self.artifact / file_name).read_text())


ALL = [citation(name) for name in NAMES]


# --- All k manifests are expected, checked and recorded -----------------------


@pytest.mark.parametrize("lead", NAMES)
def test_every_unit_is_recorded(tmp_path, lead):
    packaging = Packaging(tmp_path)
    # The operator's order does not matter; the record is in path order.
    completed = packaging.run(list(reversed(ALL)), lead=lead)
    assert completed.returncode == 0, completed.stderr
    assert packaging.inventory() == [packaging.item(name) for name in NAMES]
    files = {
        name: (
            "context-manifest.json"
            if name == lead
            else f"promotion-unit-{index:02d}-context-manifest.json"
        )
        for index, name in enumerate(NAMES)
    }
    for name, file_name in files.items():
        assert packaging.evidence(file_name) == {
            "schema": "axiom-encode/reviewed-candidate-promotion-evidence/v1",
            "citation": citation(name),
            "rulespec_path": primary(name),
        }
    assert sorted(path.name for path in packaging.artifact.iterdir()) == sorted(
        [*files.values(), "apply-manifests.json"]
    )


def test_a_batch_of_one_is_recorded_as_a_single_unit_run_is(tmp_path):
    batch = Packaging(tmp_path / "batch", names=("a",))
    assert batch.run([citation("a")]).returncode == 0
    alone = Packaging(tmp_path / "alone", names=("a",))
    assert alone.run(None, switch_on=False).returncode == 0
    assert (batch.artifact / "context-manifest.json").read_bytes() == (
        alone.artifact / "context-manifest.json"
    ).read_bytes()
    # The two checkouts differ in their commit, so in the manifest digest only.
    assert batch.inventory() == [batch.item("a")]
    assert alone.inventory() == [alone.item("a")]
    assert {**batch.item("a"), "sha256": ""} == {**alone.item("a"), "sha256": ""}
    assert sorted(path.name for path in batch.artifact.iterdir()) == [
        "apply-manifests.json",
        "context-manifest.json",
    ]


# --- Fail closed --------------------------------------------------------------


def refused(completed, message: str) -> None:
    assert completed.returncode != 0
    assert message in completed.stderr, completed.stderr


def test_a_manifest_of_an_unnamed_unit_fails_the_run(tmp_path):
    packaging = Packaging(tmp_path)
    refused(
        packaging.run(ALL[:2]),
        "multi-unit promotion changed a manifest that belongs to no named unit",
    )
    assert not (packaging.artifact / "apply-manifests.json").exists()


def test_a_named_unit_without_its_manifest_fails_the_run(tmp_path):
    packaging = Packaging(tmp_path, names=("a", "b"))
    refused(packaging.run(ALL), "lacks the manifest of a named unit")
    assert not (packaging.artifact / "apply-manifests.json").exists()


def test_a_manifest_outside_a_unit_position_fails_the_run(tmp_path):
    packaging = Packaging(tmp_path)
    packaging.write("c", at="other")
    refused(packaging.run(ALL), "belongs to no named unit")


@pytest.mark.parametrize("wrong", ["us/statute/7/2015/b", "us/statute/9/9"])
def test_a_manifest_with_another_citation_fails_the_run(tmp_path, wrong):
    """At a unit's position, but it names another unit or no unit."""
    packaging = Packaging(tmp_path)
    packaging.write("c", citation=wrong)
    completed = packaging.run(ALL)
    assert completed.returncode != 0
    assert "expected exactly one changed signed apply manifest for" in completed.stderr
    assert not (packaging.artifact / "apply-manifests.json").exists()


def test_two_manifests_at_each_others_position_fail_the_run(tmp_path):
    """Each manifest is right in itself; each sits where the other belongs."""
    packaging = Packaging(tmp_path)
    packaging.write("b", at="c")
    packaging.write("c", at="b")
    completed = packaging.run(ALL)
    assert completed.returncode != 0
    assert "does not cover exactly the pair of its unit" in completed.stderr
    assert not (packaging.artifact / "apply-manifests.json").exists()


def test_a_manifest_of_another_schema_fails_the_run(tmp_path):
    packaging = Packaging(tmp_path)
    packaging.write("b", schema_version="axiom-encode/applied-rulespec/v4")
    refused(
        packaging.run(ALL),
        f"expected exactly one changed signed apply manifest for {citation('b')}; "
        "found 0",
    )


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize(
    "change",
    [
        {"tool": "axiom-encode encode --apply"},
        {"reviewed_rulespec_ref": "f" * 40},
        {"context_manifest_file": "/tmp/context.json"},
        {"context_manifest_sha256": "0" * 64},
    ],
)
def test_every_unit_gets_the_promotion_evidence_check(tmp_path, name, change):
    """Not the unit of the citation input only."""
    packaging = Packaging(tmp_path)
    packaging.write(name, **change)
    completed = packaging.run(ALL)
    assert completed.returncode != 0
    assert (
        "reviewed candidate promotion evidence does not bind the requested RuleSpec ref"
        in completed.stderr
        or "manifest-only refresh has unexpected generation evidence"
        in completed.stderr
    ), completed.stderr
    assert not (packaging.artifact / "apply-manifests.json").exists()


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize(
    "files",
    [
        [{"path": primary("a")}, {"path": primary("b")}],
        [{"path": "us/statutes/7/2015/zz.yaml"}],
        [{"path": primary("X")}, {"path": "us/statutes/7/2015/X.test.yaml"}],
        "pair-and-extra",
        "pair-and-non-mapping",
        "primary-only",
        "not-a-list",
    ],
)
def test_every_manifest_covers_exactly_the_pair_of_its_unit(tmp_path, name, files):
    packaging = Packaging(tmp_path)
    pair = [
        {"path": primary(name)},
        {"path": primary(name).removesuffix(".yaml") + ".test.yaml"},
    ]
    if files == "pair-and-extra":
        files = [*pair, {"path": "README.md"}]
    elif files == "pair-and-non-mapping":
        files = [*pair, "README.md"]
    elif files == "primary-only":
        files = pair[:1]
    elif files == "not-a-list":
        files = {"path": primary(name)}
    packaging.write(name, applied_files=files)
    completed = packaging.run(ALL)
    assert completed.returncode != 0
    assert (
        f"does not cover exactly the pair of its unit {citation(name)}"
        in completed.stderr
        or "malformed applied_files" in completed.stderr
    ), completed.stderr
    assert not (packaging.artifact / "apply-manifests.json").exists()


def test_a_deleted_manifest_fails_the_run(tmp_path):
    packaging = Packaging(tmp_path)
    old = packaging.rulespec / ".axiom/encoding-manifests/us/statutes/7/1/old.json"
    old.parent.mkdir(parents=True)
    old.write_text("{}\n")
    subprocess.run(["git", "add", "-f", str(old)], cwd=packaging.rulespec, check=True)
    subprocess.run(
        ["git", "commit", "-qm", "old manifest"], cwd=packaging.rulespec, check=True
    )
    packaging.ref = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=packaging.rulespec, text=True
    ).strip()
    for name in NAMES:
        packaging.write(name)
    old.unlink()
    refused(packaging.run(ALL), "belongs to no named unit")


def test_the_citation_input_must_be_a_unit(tmp_path):
    packaging = Packaging(tmp_path)
    refused(
        packaging.run(ALL[1:], lead="a"),
        "must name the citation input as one of its units",
    )


def test_units_cannot_mix_with_another_lane(tmp_path):
    packaging = Packaging(tmp_path)
    (tmp_path / "source-bundle.json").write_text('["us/statute/7/1"]\n')
    refused(packaging.run(ALL), "multi-unit promotion cannot mix with other lanes")


def test_a_unit_list_with_the_switch_off_fails_the_run(tmp_path):
    packaging = Packaging(tmp_path)
    refused(
        packaging.run(ALL, switch_on=False),
        "reviewed PR promotion requires exactly transaction and rulespec_pr",
    )
    assert not (packaging.artifact / "apply-manifests.json").exists()


# --- A single-unit run is unchanged -------------------------------------------


def test_a_single_unit_run_records_what_it_recorded_before(tmp_path):
    packaging = Packaging(tmp_path, names=("a",))
    completed = packaging.run(None, switch_on=False)
    assert completed.returncode == 0, completed.stderr
    assert (packaging.artifact / "apply-manifests.json").read_text() == (
        json.dumps(
            {
                "schema": "axiom-encode/applied-manifest-inventory/v1",
                "items": [packaging.item("a")],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    assert (packaging.artifact / "context-manifest.json").read_text() == (
        json.dumps(
            {
                "schema": "axiom-encode/reviewed-candidate-promotion-evidence/v1",
                "citation": citation("a"),
                "rulespec_path": primary("a"),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    assert sorted(path.name for path in packaging.artifact.iterdir()) == [
        "apply-manifests.json",
        "context-manifest.json",
    ]


@pytest.mark.parametrize("switch_on", [False, True])
def test_a_single_unit_run_still_ignores_a_manifest_of_another_citation(
    tmp_path, switch_on
):
    """Unchanged behaviour of the single-unit form, pinned, with either value.

    Only a run that names its units fails closed on such a manifest.
    """
    packaging = Packaging(tmp_path, names=("a", "b"))
    completed = packaging.run(None, switch_on=switch_on)
    assert completed.returncode == 0, completed.stderr
    assert packaging.inventory() == [packaging.item("a")]


# --- metadata.json -------------------------------------------------------------


def run_metadata(tmp_path: Path, units, *, switch_on=True):
    heads = {
        name: init(tmp_path / name)
        for name in (
            "axiom-encode",
            "axiom-corpus",
            "axiom-rules-engine",
            "rulespec-us",
        )
    }
    runner_temp = tmp_path / "runner-temp"
    runner_temp.mkdir()
    for name, text in (
        ("source-bundle.json", "[]\n"),
        ("canonical-refresh-bundle.json", "[]\n"),
        ("existing-signed-imports.json", "[]\n"),
        ("existing-signed-import-inventory.json", "{}\n"),
    ):
        (runner_temp / name).write_text(text)
    target = tmp_path / "metadata.json"
    completed = subprocess.run(
        [sys.executable, "-", str(target)],
        cwd=tmp_path,
        input=metadata_script(),
        check=False,
        capture_output=True,
        text=True,
        env=child_env(
            tmp_path,
            switch_on=switch_on,
            CITATION=citation("a"),
            GITHUB_RUN_ATTEMPT="1",
            GITHUB_RUN_ID="5678",
            PR_BASE_BRANCH="codex/candidate",
            REPLACE_RULESPEC_PATH=primary("a"),
            RULESPEC_CHECKOUT="rulespec-us",
            RULESPEC_REF=heads["rulespec-us"],
            RUNNER_TEMP=str(runner_temp),
            ATOMIC_SOURCE_JSON=wrapper(units),
        ),
    )
    return completed, target


def test_metadata_names_every_unit(tmp_path):
    completed, target = run_metadata(tmp_path, ALL)
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(target.read_text())
    assert payload["promotion_units"] == ALL
    assert payload["citation"] == citation("a")


def test_metadata_of_a_single_unit_run_has_no_new_key(tmp_path):
    completed, target = run_metadata(tmp_path, None, switch_on=False)
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(target.read_text())
    assert "promotion_units" not in payload
    assert list(payload)[0] == "atomic_source_input"


def test_metadata_refuses_a_unit_list_with_the_switch_off(tmp_path):
    completed, target = run_metadata(tmp_path, ALL, switch_on=False)
    assert completed.returncode != 0
    assert not target.exists()


# --- Commit message, pull request title and body ------------------------------


needs_shell = pytest.mark.skipif(
    shutil.which("jq") is None or shutil.which("bash") is None,
    reason="jq and bash are required to execute the workflow fragments",
)


def fragment(run: str, start: str, end: str) -> str:
    """The lines of a step from the one that holds ``start`` to ``end``."""
    lines = run.splitlines()
    first = next(index for index, line in enumerate(lines) if start in line)
    last = next(
        index for index, line in enumerate(lines) if index > first and end in line
    )
    return "\n".join(lines[first : last + 1])


@needs_shell
@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (
            {"promotion_units": ALL},
            "Promote reviewed candidates for us/statute/7/2015/a, "
            "us/statute/7/2015/b, us/statute/7/2015/c without regeneration",
        ),
        ({}, "Promote reviewed candidate for us/statute/7/2015/a without regeneration"),
        (
            {"promotion_units": []},
            "Promote reviewed candidate for us/statute/7/2015/a without regeneration",
        ),
    ],
)
def test_the_commit_message_names_every_unit(tmp_path, payload, message):
    script = fragment(
        step("Encode, review, validate, and apply")["run"],
        'promotion_units_label="$(jq -r',
        "Promote reviewed candidate for ${CITATION} without regeneration",
    )
    completed = subprocess.run(
        [
            "bash",
            "-c",
            'set -euo pipefail\nqueue_operation() { printf "%s\\n" "$@"; }\n'
            + script
            + "\nfi\n",
        ],
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "CITATION": citation("a"),
            "atomic_source_payload": json.dumps(payload),
        },
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.splitlines() == ["checkpoint", message]


def test_the_single_unit_commit_message_line_is_unchanged():
    run = step("Encode, review, validate, and apply")["run"]
    lines = [line.strip() for line in run.splitlines()]
    message = '"Promote reviewed candidate for ${CITATION} without regeneration"'
    assert lines.count(message) == 1
    assert lines[lines.index(message) - 1] == "queue_operation checkpoint \\"


def prepare_units(tmp_path: Path, source: str, *, switch_on: bool):
    script = fragment(
        step("Prepare reviewed lane pull request branch")["run"],
        'promotion_units=""',
        "promotion_unit_count=\"$(jq -r 'length'",
    )
    (tmp_path / "axiom-encode").symlink_to(ROOT, target_is_directory=True)
    return subprocess.run(
        [
            "bash",
            "-c",
            "set -euo pipefail\nworkflow_python=("
            + sys.executable
            + ")\n"
            + script
            + '\nfi\nprintf "%s|%s\\n" "$promotion_units" "$promotion_unit_count"\n',
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        env=child_env(tmp_path, switch_on=switch_on, ATOMIC_SOURCE_JSON=source),
    )


@needs_shell
def test_the_publication_step_reads_the_units_without_a_token(tmp_path):
    completed = prepare_units(tmp_path, wrapper(ALL), switch_on=True)
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == ", ".join(ALL) + "|3\n"
    preparation = step("Prepare reviewed lane pull request branch")
    publication = step("Push lane branch and open draft pull request")
    assert "secrets." not in str(preparation.get("env", {}))
    assert (
        preparation["env"]["ATOMIC_SOURCE_JSON"] == "${{ inputs.source_bundle_json }}"
    )
    assert 'echo "promotion_units=$promotion_units"' in preparation["run"]
    assert 'echo "promotion_unit_count=$promotion_unit_count"' in preparation["run"]
    outputs = "${{ steps.reviewed_lane_pull_request_branch.outputs."
    assert publication["env"]["PROMOTION_UNITS"] == outputs + "promotion_units }}"
    assert publication["env"]["PROMOTION_UNIT_COUNT"] == (
        outputs + "promotion_unit_count }}"
    )
    # The token-bearing step never reads the dispatch input itself.
    assert "ATOMIC_SOURCE_JSON" not in str(publication)


@needs_shell
@pytest.mark.parametrize("source", [wrapper(None), "[]", '{"schema":"other"}', ""])
def test_a_run_without_units_publishes_no_unit_list(tmp_path, source):
    completed = prepare_units(tmp_path, source, switch_on=False)
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == "|0\n"


@needs_shell
@pytest.mark.parametrize("unit", ["x`id`", "a b", "a, b", "$(id)", "a'b", 'a"b'])
def test_a_unit_that_is_not_a_plain_citation_stops_the_publication(tmp_path, unit):
    """The title and body take the units only as plain citation text."""
    completed = prepare_units(tmp_path, wrapper([citation("a"), unit]), switch_on=True)
    assert completed.returncode != 0
    assert "|" not in completed.stdout


@needs_shell
def test_a_unit_list_with_the_switch_off_stops_the_publication(tmp_path):
    completed = prepare_units(tmp_path, wrapper(ALL), switch_on=False)
    assert completed.returncode != 0
    assert "|" not in completed.stdout


def publish_title_and_body(units: str, count: str) -> list[str]:
    script = fragment(
        step("Push lane branch and open draft pull request")["run"],
        'title="Add signed encoding manifest for ${CITATION}"',
        "fi",
    )
    completed = subprocess.run(
        [
            "bash",
            "-c",
            'set -euo pipefail\nbody="existing body"\n'
            + script
            + '\nprintf "%s\\n" "$title" "$body"\n',
        ],
        capture_output=True,
        text=True,
        check=True,
        env={
            **os.environ,
            "CITATION": citation("a"),
            "PROMOTION_UNITS": units,
            "PROMOTION_UNIT_COUNT": count,
        },
    )
    return completed.stdout.splitlines()


@needs_shell
def test_the_pull_request_title_and_body_name_every_unit():
    assert publish_title_and_body(", ".join(ALL), "3") == [
        "Add signed encoding manifests for 3 units (us/statute/7/2015/a and others)",
        "existing body",
        "",
        "Units (3): `us/statute/7/2015/a, us/statute/7/2015/b, us/statute/7/2015/c`",
    ]


@needs_shell
def test_the_single_unit_pull_request_title_and_body_are_unchanged():
    assert publish_title_and_body("", "0") == [
        "Add signed encoding manifest for us/statute/7/2015/a",
        "existing body",
    ]
    # The text a single-unit run always had, still in the step.
    assert (
        'title="Add signed encoding manifest for ${CITATION}"'
        in step("Push lane branch and open draft pull request")["run"]
    )
    publication = step("Push lane branch and open draft pull request")["run"]
    assert '-f title="$title" \\' in publication
    assert publication.count("gh api --method POST") == 1
