"""Content-keyed packaging of validate-rulespec pins (ci_parity_pins).

Invariants, each checked for every packaged pin or every generated input:

1. Derivation: a pin's inputs, gate order, gate registry (flags and line
   ranges), blob and gate parameters derived from its packaged workflow equal
   the hand tables ci held before, for every pin those tables covered.
2. Reviewed glue is exactly the glue units of the template pins, so a change
   to the extraction itself fails here before it can widen admission.
3. Every packaged pin that executes its own scripts is admitted, with the
   features its review established; ff774c29 and df2dfb53 were packaged by
   ``ci-pin add`` alone.
4. Digests: editing an exit-status program keeps its step's digest exactly
   when the program's interface is unchanged; editing any glue, or an
   output-consumed program, always changes it (or is refused).
5. Admission refuses unreviewed glue, other step orders, partial feature
   groups, conflicting variants, producer/consumer pairs no reviewed pin has,
   missing inputs, duplicate or unnamed steps, and programs ci and bash could
   read differently.
6. ``ci-pin add`` packages byte-exact from a verified .github main commit, is
   idempotent, and writes nothing when it refuses.
"""

from __future__ import annotations

import ast
import copy
import io
import json
import os
import re
import shutil
import subprocess
from argparse import Namespace
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import yaml
from hypothesis import HealthCheck, event, given, settings
from hypothesis import strategies as st

from axiom_encode import ci_parity, ci_parity_pins
from axiom_encode.ci_parity import (
    DEPENDENCY_INPUTS,
    SUPPORTED_WORKFLOW_PINS,
    WORKFLOW_DIRECTORY,
    WORKFLOW_ENVIRONMENT_STEPS,
    WORKFLOW_GATE_STEPS,
    WORKFLOW_RESOLUTION_STEPS,
    PinnedWorkflow,
)
from axiom_encode.ci_parity_pins import (
    CI_READ_INPUTS,
    COMPOSER_INPUT,
    EXECUTED_PROGRAMS,
    FEATURE_SCOPES,
    FEATURE_STEPS,
    FEATURES,
    LEGACY_PINS,
    MANIFEST_NAME,
    PROGRAM_MARKER,
    REVIEWED_GLUE_NAME,
    TEMPLATE_FEATURES,
    TEMPLATE_PINS,
    GlueError,
    PinSourceError,
    add_pin,
    admit,
    declared_inputs,
    executed_program,
    git_blob_id,
    glue_digest,
    glue_units,
    load_manifest,
    load_reviewed_glue,
    packaged_admission,
    packaged_bytes,
    packaged_payload,
    program_interface,
    render_report,
    review_context,
    run_ci_pin,
    step_glue_view,
    step_spans,
    template_glue,
    write_reviewed_glue,
)
from tests.ci_parity_hand_tables import PINS as HAND_PINS

PIN_0EFFA6A5 = "0effa6a5b05e7fac53902df7d523e909bd7fc48a"
PIN_6F11BE26 = "6f11be2655f79dd0a3b582db46525f58332ca120"
PIN_92CCB0F2 = "92ccb0f2266bf06aba6111d2bfd0e5f03af55fc3"
PIN_FF774C29 = "ff774c29a58448592c3ac92c9195d4995f5f4b57"
PIN_DF2DFB53 = "df2dfb5310c6bc8d71abdb69281a7c71afc9663e"
EMBEDDED = tuple(
    sha for sha in sorted(SUPPORTED_WORKFLOW_PINS) if sha not in LEGACY_PINS
)
PRECHECK = ("shards", "Reject unmanifested RuleSpec content")
WORKFLOW_PATH = ".github/workflows/validate-rulespec.yml"
GITHUB_URL = "https://github.com/TheAxiomFoundation/.github.git"


def _payload(sha: str) -> dict[str, Any]:
    return copy.deepcopy(packaged_payload(sha))


def _step(payload: dict[str, Any], job: str, name: str) -> dict[str, Any]:
    return next(step for step in payload["jobs"][job]["steps"] if step["name"] == name)


def _unit(payload: dict[str, Any], job: str, name: str) -> ci_parity_pins.GlueUnit:
    return next(
        unit
        for unit in glue_units(payload)
        if unit.kind == "step" and (unit.job, unit.step) == (job, name)
    )


def _admit(payload: dict[str, Any]) -> ci_parity_pins.Admission:
    return admit(payload, review_context())


def _kinds(admission: ci_parity_pins.Admission) -> set[str]:
    return {problem.kind for problem in admission.problems}


# ---------------------------------------------------------------------------
# 1. Derivation reproduces the hand tables
# ---------------------------------------------------------------------------

# The hand tables ended 6f11be26's and 92ccb0f2's changed-oracle gate on the
# two comment lines that introduce "Run repository tests"; 0effa6a5's ended at
# the step's last line. Derivation ends all three at the PY terminator.
LINE_CORRECTIONS = {
    (PIN_6F11BE26, "changed_oracle_coverage"): "2513-2617",
    (PIN_92CCB0F2, "changed_oracle_coverage"): "3109-3227",
}


@pytest.mark.parametrize("sha", sorted(HAND_PINS), ids=lambda sha: sha[:8])
def test_derived_pins_reproduce_the_hand_tables(sha: str) -> None:
    pin = SUPPORTED_WORKFLOW_PINS[sha]
    hand = HAND_PINS[sha]

    assert pin.fixture == hand["fixture"]
    assert pin.workflow_blob == hand["workflow_blob"]
    assert list(pin.inputs.items()) == list(hand["inputs"].items())
    assert pin.gates == hand["gates"]
    assert pin.gate_parameters == hand["gate_parameters"]
    assert pin.registry == tuple(
        replace(
            spec,
            workflow_lines=LINE_CORRECTIONS.get((sha, spec.key), spec.workflow_lines),
        )
        for spec in hand["registry"]
    )


@pytest.mark.parametrize(("sha", "key"), sorted(LINE_CORRECTIONS))
def test_corrected_line_ranges_end_at_the_programs_terminator(
    sha: str, key: str
) -> None:
    lines = (
        (WORKFLOW_DIRECTORY / SUPPORTED_WORKFLOW_PINS[sha].fixture)
        .read_text()
        .split("\n")
    )
    end = int(LINE_CORRECTIONS[(sha, key)].split("-")[1])

    assert lines[end - 1].strip() == "PY"
    assert lines[end].strip() == ""
    assert lines[end + 1].lstrip().startswith("# Repository-controlled tests")
    assert lines[end + 3].lstrip() == "- name: Run repository tests"


@pytest.mark.parametrize("sha", EMBEDDED, ids=lambda sha: sha[:8])
def test_line_ranges_start_and_end_on_their_steps(sha: str) -> None:
    text = (WORKFLOW_DIRECTORY / SUPPORTED_WORKFLOW_PINS[sha].fixture).read_text()
    lines = text.split("\n")
    spans = step_spans(text)
    for spec in SUPPORTED_WORKFLOW_PINS[sha].registry:
        start, end = (int(part) for part in spec.workflow_lines.split("-"))
        covered = [
            span
            for (_, name), span in spans.items()
            if spec.key in WORKFLOW_GATE_STEPS.get(name, ())
        ]
        assert start == min(first for first, _ in covered)
        assert end == max(last for _, last in covered)
        assert lines[start - 1].lstrip().startswith("- name: ")
        assert lines[end - 1].strip()


def _step_columns(text: str) -> dict[int, int]:
    """1-based first line of each step mapping -> the mapping's column."""

    root = yaml.compose(text, Loader=yaml.SafeLoader)
    jobs = next(value for key, value in root.value if key.value == "jobs")
    return {
        step.start_mark.line + 1: step.start_mark.column
        for _, job in jobs.value
        for key, steps in job.value
        if key.value == "steps"
        for step in steps.value
    }


@pytest.mark.parametrize("sha", EMBEDDED, ids=lambda sha: sha[:8])
def test_step_spans_start_on_each_step_and_leave_no_step_content_outside(
    sha: str,
) -> None:
    text = (WORKFLOW_DIRECTORY / SUPPORTED_WORKFLOW_PINS[sha].fixture).read_text()
    lines = text.split("\n")
    columns = _step_columns(text)
    assert set(columns.values()) == {8}
    spans = sorted(step_spans(text).values())

    assert [start for start, _ in spans] == sorted(columns)
    following = [start for start, _ in spans[1:]] + [len(lines) + 1]
    for (start, end), next_start in zip(spans, following, strict=True):
        assert start <= end < next_start
        between = lines[end : next_start - 1]
        if any(re.match(r"^  [A-Za-z0-9_-]+:", line) for line in between):
            continue  # the next step is in the next job
        # Within a job only blank lines and the next step's comments lie
        # between two spans: no step content is left out of one.
        assert all(
            not line.strip() or line.lstrip().startswith("#") for line in between
        ), (sha[:8], start, between)


@pytest.mark.parametrize(
    "sha", sorted(SUPPORTED_WORKFLOW_PINS), ids=lambda sha: sha[:8]
)
def test_manifest_blobs_are_the_packaged_bytes(sha: str) -> None:
    manifest = load_manifest()
    data = (WORKFLOW_DIRECTORY / manifest[sha].file).read_bytes()

    assert git_blob_id(data) == manifest[sha].blob
    assert packaged_bytes(sha) == data


def test_manifest_lists_every_packaged_workflow() -> None:
    files = {path.name for path in WORKFLOW_DIRECTORY.glob("validate-rulespec-*.yml")}

    assert files == {pin.file for pin in load_manifest().values()}
    assert (WORKFLOW_DIRECTORY / MANIFEST_NAME).read_text().endswith("}\n")


def test_tampered_packaged_workflow_is_refused(tmp_path: Path) -> None:
    shutil.copytree(WORKFLOW_DIRECTORY, tmp_path / "workflows")
    directory = tmp_path / "workflows"
    pin = load_manifest(directory)[PIN_FF774C29]
    path = directory / pin.file
    path.write_bytes(path.read_bytes() + b"\n")

    with pytest.raises(ValueError, match="is blob"):
        ci_parity_pins.read_packaged(pin, directory)


# ---------------------------------------------------------------------------
# 2. Reviewed glue is the template pins' glue
# ---------------------------------------------------------------------------


def test_reviewed_glue_is_every_template_unit_and_nothing_else(tmp_path: Path) -> None:
    entries = load_reviewed_glue()
    regenerated = template_glue(
        {sha: packaged_payload(sha) for sha in TEMPLATE_PINS}, entries
    )

    assert set(entries) == set(regenerated)
    write_reviewed_glue(regenerated, tmp_path)
    assert (tmp_path / REVIEWED_GLUE_NAME).read_bytes() == (
        WORKFLOW_DIRECTORY / REVIEWED_GLUE_NAME
    ).read_bytes()


def test_only_the_composer_variants_carry_a_flag_profile() -> None:
    flagged = {
        (entry.job, entry.step, entry.reviewed_in)
        for entry in load_reviewed_glue()
        if entry.gate_flags is not None
    }

    assert flagged == {
        ("validate", "Enforce validation waiver ratchet", "92ccb0f2"),
        ("validate", "Validate RuleSpec YAML", "92ccb0f2"),
        ("validate", "Execute RuleSpec companion tests", "92ccb0f2"),
    }
    assert set(ci_parity._WORKFLOW_GATE_FLAG_PROFILES) == {"composer"}


def test_a_changed_extraction_fails_the_review_context() -> None:
    templates = {sha: packaged_payload(sha) for sha in TEMPLATE_PINS}
    entries = list(load_reviewed_glue())
    dropped = entries.pop()

    with pytest.raises(ValueError, match="does not list"):
        ci_parity_pins.build_review_context(templates, entries)
    with pytest.raises(ValueError, match="no template has"):
        ci_parity_pins.build_review_context(
            templates, [*entries, dropped, replace(dropped, digest="0" * 64)]
        )
    with pytest.raises(ValueError, match="first occurs"):
        ci_parity_pins.build_review_context(
            templates, [*entries, replace(dropped, reviewed_in="ffffffff")]
        )


def test_every_template_step_is_classified_exactly_once() -> None:
    for sha in TEMPLATE_PINS:
        for unit in glue_units(packaged_payload(sha)):
            if unit.kind != "step":
                continue
            roles = (
                (unit.step in WORKFLOW_GATE_STEPS)
                + (unit.step in WORKFLOW_RESOLUTION_STEPS)
                + (unit.step in WORKFLOW_ENVIRONMENT_STEPS)
            )
            assert roles == 1, unit.label


def test_executed_programs_occur_in_the_templates_and_extract_cleanly() -> None:
    present = {
        (unit.job, unit.step)
        for sha in TEMPLATE_PINS
        for unit in glue_units(packaged_payload(sha))
        if unit.kind == "step"
    }

    assert set(EXECUTED_PROGRAMS) <= present
    for sha in EMBEDDED:
        workflow = PinnedWorkflow(sha)
        for (job, name), program in EXECUTED_PROGRAMS.items():
            if not any(
                step.get("name") == name
                for step in workflow.payload["jobs"][job]["steps"]
            ):
                continue
            read = (
                workflow.python
                if program.embedding == "heredoc"
                else workflow.inline_python
            )
            source = read(job, name)
            assert "${{" not in source
            ast.parse(source)


def test_pinned_workflow_refuses_programs_ci_does_not_execute() -> None:
    workflow = PinnedWorkflow(PIN_92CCB0F2)

    with pytest.raises(ValueError, match="no heredoc program"):
        workflow.python("validate", "Install changed-file oracle coverage classifier")
    with pytest.raises(ValueError, match="no inline program"):
        workflow.inline_python("validate", "Validate RuleSpec YAML")


# Each feature's scope is a review claim: nothing outside it reads the
# feature. These are the claims' witnesses in the templates' glue.
SCOPE_WITNESSES = {
    "isolated_dependencies": ("AXIOM_DEPENDENCY_", "_axiom/"),
    "composer_runtime": (
        "axiom-compose-ref",
        "AXIOM_COMPOSE_REF",
        "composer_args",
        "axiom-compose-verification",
    ),
    "program_spec_precheck": ("ProgramSpec", "shape-check"),
    "exact_guard_base_ref": ("generated-guard-base-ref", "GeneratedGuardBaseRef"),
    "retired_corpus_release_key": ("RETIRED_CORPUS_RELEASE", "RETIRED_PUBLIC_KEY"),
    "unmanifested_precheck": ("SCAN_MODE", "unmanifested"),
}


@pytest.mark.parametrize("feature", sorted(SCOPE_WITNESSES))
def test_feature_witnesses_stay_inside_the_features_scope(feature: str) -> None:
    scope = FEATURE_SCOPES[feature]
    inside: set[str] = set()
    for sha in TEMPLATE_PINS:
        for unit in glue_units(packaged_payload(sha)):
            text = ci_parity_pins.canonical_json(unit.view)
            for witness in SCOPE_WITNESSES[feature]:
                if scope.contains(unit):
                    inside.update({witness} if witness in text else set())
                else:
                    assert witness not in text, (sha[:8], unit.label, witness)
    # Each witness is the feature's own text: it occurs inside the scope.
    assert inside == set(SCOPE_WITNESSES[feature])


def test_every_feature_has_a_scope_and_group_features_match_template_presence() -> None:
    assert set(FEATURE_SCOPES) == set(FEATURES)
    assert set(FEATURE_STEPS) <= set(FEATURES)
    for sha, features in TEMPLATE_FEATURES.items():
        names = {
            (job, name)
            for job, names in ci_parity_pins.job_shape(packaged_payload(sha))
            for name in names
        }
        for feature, steps in FEATURE_STEPS.items():
            assert features[feature] is (steps <= names), (sha[:8], feature)
            assert features[feature] or not (steps & names), (sha[:8], feature)


def test_every_input_ci_reads_is_one_admission_requires() -> None:
    source = Path(ci_parity.__file__).read_text()
    read = set(re.findall(r'inputs\["([a-z0-9-]+)"\]', source)) | set(
        re.findall(r'inputs\.get\(\s*"([a-z0-9-]+)"', source)
    )

    assert read
    assert read <= set(CI_READ_INPUTS) | {COMPOSER_INPUT}
    assert set(DEPENDENCY_INPUTS.values()) <= set(CI_READ_INPUTS)
    assert set(ci_parity.OPTIONAL_DEPENDENCY_INPUTS.values()) == {COMPOSER_INPUT}


# ---------------------------------------------------------------------------
# 3. Packaged pins are admitted
# ---------------------------------------------------------------------------

EXPECTED_ADMISSION = {
    PIN_0EFFA6A5: (PIN_0EFFA6A5, ()),
    PIN_6F11BE26: (PIN_6F11BE26, ()),
    PIN_92CCB0F2: (PIN_92CCB0F2, ()),
    PIN_FF774C29: (PIN_92CCB0F2, ()),
    PIN_DF2DFB53: (PIN_92CCB0F2, ("program_spec_precheck",)),
}


def test_every_embedded_pin_is_known_to_this_table() -> None:
    assert set(EMBEDDED) == set(EXPECTED_ADMISSION)
    assert set(TEMPLATE_PINS) <= set(EMBEDDED)


@pytest.mark.parametrize("sha", sorted(EXPECTED_ADMISSION), ids=lambda sha: sha[:8])
def test_packaged_pins_are_admitted_with_their_features(sha: str) -> None:
    admission = packaged_admission(sha)
    template, removed = EXPECTED_ADMISSION[sha]
    expected = dict(TEMPLATE_FEATURES[template])
    for group in removed:
        expected[group] = False

    assert admission.problems == []
    assert (admission.template, admission.removed_groups) == (template, removed)
    assert admission.features == expected
    pin = SUPPORTED_WORKFLOW_PINS[sha]
    assert pin.gate_parameters == ci_parity.WorkflowGateParameters(
        exclude_programs_from_money_atom_check=True, embedded_scripts=True, **expected
    )


def test_ff774c29_is_92ccb0f2_but_for_the_precheck_program() -> None:
    old, new = _payload(PIN_92CCB0F2), _payload(PIN_FF774C29)
    old_run, new_run = _step(old, *PRECHECK)["run"], _step(new, *PRECHECK)["run"]
    body = ci_parity_pins.heredoc_bodies
    _step(new, *PRECHECK)["run"] = new_run.replace(body(new_run)[0], body(old_run)[0])

    assert body(new_run)[0] != body(old_run)[0]
    assert new == old
    assert EXECUTED_PROGRAMS[PRECHECK].consumes == "exit-status"
    old_pin, new_pin = (
        SUPPORTED_WORKFLOW_PINS[PIN_92CCB0F2],
        SUPPORTED_WORKFLOW_PINS[PIN_FF774C29],
    )
    assert new_pin.inputs == old_pin.inputs
    assert new_pin.gates == old_pin.gates
    assert [(s.key, s.name, s.subcommand, s.flags) for s in new_pin.registry] == [
        (s.key, s.name, s.subcommand, s.flags) for s in old_pin.registry
    ]


def test_df2dfb53_is_92ccb0f2_without_the_parser_and_with_6f11be26s_precheck() -> None:
    expected = _payload(PIN_92CCB0F2)
    parser = {name for _, name in FEATURE_STEPS["program_spec_precheck"]}
    old_precheck = _step(_payload(PIN_6F11BE26), *PRECHECK)
    expected["jobs"]["shards"]["steps"] = [
        old_precheck if step["name"] == PRECHECK[1] else step
        for step in expected["jobs"]["shards"]["steps"]
        if step["name"] not in parser
    ]
    actual = _payload(PIN_DF2DFB53)
    for payload in (expected, actual):
        for declaration in payload[True]["workflow_call"]["inputs"].values():
            declaration.pop("description", None)

    assert actual == expected
    pin = SUPPORTED_WORKFLOW_PINS[PIN_DF2DFB53]
    assert pin.gates == SUPPORTED_WORKFLOW_PINS[PIN_92CCB0F2].gates
    # The 6f11be26 pre-check declares no GUARD_PROGRAMS_ROOT, so ci passes none.
    assert "GUARD_PROGRAMS_ROOT" not in _step(actual, *PRECHECK)["env"]
    assert ci_parity.program_spec_parser_mismatch(PinnedWorkflow(PIN_DF2DFB53)) is None


# ---------------------------------------------------------------------------
# 4. Digest properties
# ---------------------------------------------------------------------------

_PAYLOAD_92CC = packaged_payload(PIN_92CCB0F2)
_INPUTS_92CC = declared_inputs(_PAYLOAD_92CC)
_EXIT_STEPS = sorted(
    key
    for key, program in EXECUTED_PROGRAMS.items()
    if program.consumes == "exit-status"
)
_OUTPUT_STEPS = sorted(
    key for key, program in EXECUTED_PROGRAMS.items() if program.consumes == "output"
)
_ALL_STEPS = sorted(
    (job, step["name"])
    for job, body in _PAYLOAD_92CC["jobs"].items()
    for step in body["steps"]
)


def _digest_of(job: str, step: dict[str, Any]) -> str:
    return glue_digest(step_glue_view(job, step, _INPUTS_92CC))


def _program_span(run: str, embedding: str) -> tuple[int, int]:
    if embedding == "inline":
        return ci_parity_pins._inline_span(run)
    first, end = ci_parity_pins._heredoc_spans(run)[0]
    lines = run.split("\n")
    start = sum(len(line) + 1 for line in lines[:first])
    stop = sum(len(line) + 1 for line in lines[:end])
    return start, stop


def _top_level_offsets(source: str) -> list[int]:
    """Character offsets of the start of each top-level statement."""

    lines = source.split("\n")
    starts = [statement.lineno for statement in ast.parse(source).body]
    return [sum(len(line) + 1 for line in lines[: number - 1]) for number in starts] + [
        len(source)
    ]


_NEUTRAL_LINES = st.one_of(
    st.text(
        st.characters(blacklist_categories=("Cc", "Cs"), blacklist_characters="\n\r"),
        max_size=30,
    ).map(lambda text: f"# {text}"),
    st.integers().map(lambda number: f"_unused = {number}"),
    st.just("pass"),
    st.text("abcdefgh ", max_size=12).map(lambda text: f"_label = {text!r}"),
)
# Lines that widen what a program reads or changes; none of the names occurs
# in a packaged program, so each one must change the interface.
_INTERFACE_LINES = st.sampled_from(
    [
        'import os; os.environ["CI_PIN_PROPERTY_NEW"]',
        'import os; os.getenv("CI_PIN_PROPERTY_NEW")',
        'from os import environ as ci_pin_env; ci_pin_env.get("CI_PIN_PROPERTY_NEW")',
        'from os import getenv as ci_pin_getenv; ci_pin_getenv("CI_PIN_PROPERTY_NEW")',
        "import socket",
        'import subprocess; subprocess.run(["curl", "-s"])',
        'from subprocess import run as ci_pin_run; ci_pin_run(["curl", "-s"])',
        'open("ci-pin-property", "w")',
        'import io; io.open("ci-pin-property", "w")',
        '_path = "/opt/ci-pin-property"',
        "import os; dict(os.environ)",
        'import os; getattr(os, "environ")',
        'exec("1")',
    ]
)


@settings(max_examples=200, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    key=st.sampled_from(_EXIT_STEPS),
    neutral=st.lists(_NEUTRAL_LINES, max_size=3),
    widening=st.lists(_INTERFACE_LINES, max_size=2),
    data=st.data(),
)
def test_exit_status_program_edits_keep_the_digest_iff_the_interface_holds(
    key: tuple[str, str], neutral: list[str], widening: list[str], data: st.DataObject
) -> None:
    job, name = key
    step = _step(_PAYLOAD_92CC, job, name)
    program = EXECUTED_PROGRAMS[key]
    run = step["run"]
    start, stop = _program_span(run, program.embedding)
    source = run[start:stop]
    lines = data.draw(st.permutations([*neutral, *widening]))
    offset = data.draw(st.sampled_from(_top_level_offsets(source)))
    edited_source = (
        source[:offset] + "".join(f"{line}\n" for line in lines) + source[offset:]
    )
    edited = {**step, "run": run[:start] + edited_source + run[stop:]}

    if program.embedding == "inline" and "'" in edited_source:
        event("refused: quote in an inline program")
        with pytest.raises(GlueError, match="single quote"):
            _digest_of(job, edited)
        return
    if "${{" in edited_source:
        event("refused: expression")
        with pytest.raises(GlueError, match="expression"):
            _digest_of(job, edited)
        return
    same = _digest_of(job, edited) == _digest_of(job, step)
    # The oracle is the edit itself, not the extractor under test: only
    # comments, unused assignments and `pass` keep the digest.
    event("widening edit" if widening else "neutral edit")
    assert same is (not widening), lines
    assert (program_interface(edited_source) == program_interface(source)) is same


@pytest.mark.parametrize(
    ("line", "same"),
    [
        ("# a reworded comment", True),
        ("_unused = 1", True),
        ('import os; os.environ["CI_PIN_PROPERTY_NEW"]', False),
        ('from os import environ as e; e.get("CI_PIN_PROPERTY_NEW")', False),
        ('from os import getenv as g; g("GITHUB_BASE_REF")', False),
        ("import socket", False),
        ('open("ci-pin-property", "w")', False),
    ],
)
def test_precheck_program_edits_keep_the_digest_only_with_the_interface(
    line: str, same: bool
) -> None:
    # Both directions of the property above, on ff774c29's actual change site.
    step = _step(_PAYLOAD_92CC, *PRECHECK)
    run = step["run"]
    start, stop = _program_span(run, "heredoc")
    source = run[start:stop]
    edited = {**step, "run": run[:start] + f"{line}\n" + source + run[stop:]}

    assert (_digest_of(PRECHECK[0], edited) == _digest_of(PRECHECK[0], step)) is same


@settings(max_examples=200, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    key=st.sampled_from(_ALL_STEPS),
    character=st.characters(blacklist_categories=("Cs",), blacklist_characters="\n\r"),
    data=st.data(),
)
def test_any_glue_edit_changes_the_digest_or_is_refused(
    key: tuple[str, str], character: str, data: st.DataObject
) -> None:
    job, name = key
    step = _step(_PAYLOAD_92CC, job, name)
    original = _digest_of(job, step)
    choice = data.draw(st.sampled_from(sorted(set(step) - {"name"})))
    edited = copy.deepcopy(step)
    if choice == "run":
        run = step["run"]
        program = EXECUTED_PROGRAMS.get(key)
        forbidden: range = range(0)
        if program is not None and program.consumes == "exit-status":
            start, stop = _program_span(run, program.embedding)
            forbidden = range(start, stop + 1)
        positions = [index for index in range(len(run) + 1) if index not in forbidden]
        index = data.draw(st.sampled_from(positions))
        edited["run"] = run[:index] + character + run[index:]
    else:
        edited[choice] = {"edited": edited[choice], "by": character}
    try:
        changed = _digest_of(job, edited)
    except GlueError:
        return
    assert changed != original


@settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(key=st.sampled_from(_OUTPUT_STEPS), line=_NEUTRAL_LINES, data=st.data())
def test_output_program_edits_always_change_the_digest(
    key: tuple[str, str], line: str, data: st.DataObject
) -> None:
    job, name = key
    step = _step(_PAYLOAD_92CC, job, name)
    run = step["run"]
    start, stop = _program_span(run, EXECUTED_PROGRAMS[key].embedding)
    source = run[start:stop]
    offset = data.draw(st.sampled_from(_top_level_offsets(source)))
    edited = {
        **step,
        "run": run[:start]
        + source[:offset]
        + f"{line}\n"
        + source[offset:]
        + run[stop:],
    }
    if "${{" in line:
        with pytest.raises(GlueError):
            _digest_of(job, edited)
        return
    assert _digest_of(job, edited) != _digest_of(job, step)


@pytest.mark.parametrize("name", ["validation-workers", "python-version"])
def test_a_referenced_input_declaration_is_part_of_the_digest(name: str) -> None:
    payload = _payload(PIN_92CCB0F2)
    before = {unit.digest for unit in glue_units(payload)}
    payload[True]["workflow_call"]["inputs"][name]["default"] = "changed"
    after = {unit.digest for unit in glue_units(payload)}

    assert before - after
    readers = {unit.label for unit in glue_units(payload) if unit.digest not in before}
    assert all(
        name in json.dumps(dict(unit.view))
        for unit in glue_units(payload)
        if unit.label in readers
    )


def test_input_descriptions_are_not_glue() -> None:
    payload = _payload(PIN_92CCB0F2)
    before = [unit.digest for unit in glue_units(payload)]
    for declaration in payload[True]["workflow_call"]["inputs"].values():
        declaration["description"] = "reworded"

    assert [unit.digest for unit in glue_units(payload)] == before


# ---------------------------------------------------------------------------
# Program interface
# ---------------------------------------------------------------------------


def test_program_interface_names_what_a_program_reads_and_changes() -> None:
    interface = program_interface(
        "import os, json\n"
        "from pathlib import Path\n"
        "import subprocess as sp\n"
        "a = os.environ['ALPHA']\n"
        "b = os.environ.get('BETA', '')\n"
        "c = os.getenv('GAMMA')\n"
        "d = 'DELTA' in os.environ\n"
        "e = os.environ[a]\n"
        "sp.run(['git', 'status'])\n"
        "sp.check_output(command)\n"
        "os.system('make all')\n"
        "Path('x').write_text('y')\n"
        "open('z', 'a')\n"
        "open('r')\n"
        "'text'.replace('t', 'T')\n"
        "pattern = re.compile('x')\n"
        "p = '/opt/axiom-verification/bin'\n"
        "q = '/ is not a path'\n"
    )

    assert interface.environment == ("<dynamic>", "ALPHA", "BETA", "DELTA", "GAMMA")
    assert interface.imports == ("json", "os", "pathlib", "subprocess")
    assert interface.executables == ("<dynamic>", "git", "shell:make")
    assert interface.effects == ("open:a", "write_text")
    assert interface.paths == ("/opt/axiom-verification/bin",)
    # What cannot be named is kept verbatim, so adding another is a change.
    assert interface.opaque == (
        "dynamic-environment: os.environ[a]",
        "dynamic-executable: sp.check_output(command)",
    )
    assert interface.flags == ("dynamic-environment", "dynamic-executable")


# An ordinary rewrite must not hide an access behind another name.
ALIASED_ACCESS = [
    ("from os import environ as env\nx = env.get('NEW')\n", "environment", "NEW"),
    ("from os import environ as env\nx = env['NEW']\n", "environment", "NEW"),
    ("from os import getenv as g\nx = g('NEW')\n", "environment", "NEW"),
    ("import os as o\nx = o.getenv('NEW')\n", "environment", "NEW"),
    ("from subprocess import run as r\nr(['curl', '-s'])\n", "executables", "curl"),
    ("import subprocess as sp\nsp.Popen(['wget'])\n", "executables", "wget"),
    ("import io\nio.open('f', 'w')\n", "effects", "open:w"),
    ("import codecs\ncodecs.open('f', 'a')\n", "effects", "open:a"),
    ("from pathlib import Path\nPath('f').open('w')\n", "effects", "open:w"),
    ("import os\nos.fdopen(3, 'w')\n", "effects", "open:w"),
    (
        "import os\nos.open('f', os.O_WRONLY | os.O_CREAT)\n",
        "effects",
        "os.open:os.O_WRONLY | os.O_CREAT",
    ),
    ("import os\nos.replace('a', 'b')\n", "effects", "os.replace"),
    ("import shutil as sh\nsh.copy('a', 'b')\n", "effects", "shutil.copy"),
    (
        "import os\nx = getattr(os, 'environ')['NEW']\n",
        "opaque",
        "dynamic-code: getattr(os, 'environ')",
    ),
    (
        "import os\ng = os.getenv\nx = g('NEW')\n",
        "opaque",
        "dynamic-environment: os.getenv",
    ),
    (
        "import subprocess\nf = subprocess.run\nf(['curl'])\n",
        "opaque",
        "dynamic-executable: subprocess.run",
    ),
    (
        "import sys\nx = sys.modules['os'].environ\n",
        "opaque",
        "dynamic-code: sys.modules",
    ),
    (
        "from os import *\nx = environ['NEW']\n",
        "opaque",
        "star-import: from os import *",
    ),
]


@pytest.mark.parametrize(("source", "field", "expected"), ALIASED_ACCESS)
def test_program_interface_resolves_aliases_and_indirection(
    source: str, field: str, expected: str
) -> None:
    assert expected in getattr(program_interface(source), field)


@pytest.mark.parametrize(
    ("source", "flag"),
    [
        ("import os\nenv = dict(os.environ)\n", "environment-escapes"),
        ("import os\nsubprocess.run(['x'], env=os.environ)\n", "environment-escapes"),
        ("import os\nfor key in os.environ:\n    pass\n", "environment-escapes"),
        ("import importlib\nimportlib.import_module('x')\n", "dynamic-import"),
        ("__import__('x')\n", "dynamic-code"),
        ("exec('print(1)')\n", "dynamic-code"),
        ("eval('1')\n", "dynamic-code"),
        ("x = vars(os)['environ']\n", "dynamic-code"),
        ("x = os.__dict__['environ']\n", "dynamic-code"),
        ("from pathlib import Path\nPath.home()\n", "home-directory"),
        ("import os\nos.path.expanduser('~')\n", "home-directory"),
        ("from . import x\n", "relative-import"),
    ],
)
def test_program_interface_flags_opaque_access(source: str, flag: str) -> None:
    assert flag in program_interface(source).flags


def test_ordinary_code_is_not_opaque() -> None:
    interface = program_interface(
        "import re, json\n"
        "pattern = re.compile('[a-z]+')\n"
        "text = 'a b'.replace(' ', '')\n"
        "items = [1, 2]\n"
        "items.remove(1)\n"
        "copy = dict(a=1).copy()\n"
        "data = json.loads('{}')\n"
    )

    assert (interface.opaque, interface.effects, interface.executables) == ((), (), ())


def test_two_opaque_accesses_of_one_kind_are_two_interface_entries() -> None:
    one = program_interface("import os\na = getattr(os, 'sep')\n")
    two = program_interface(
        "import os\na = getattr(os, 'sep')\nb = getattr(os, 'environ')\n"
    )

    assert one.flags == two.flags == ("dynamic-code",)
    assert one != two


def test_environ_import_alone_is_not_an_escape() -> None:
    assert program_interface("from os import environ\nx = environ['A']\n").flags == ()


# ---------------------------------------------------------------------------
# 5. Admission refusals
# ---------------------------------------------------------------------------


def test_unreviewed_glue_is_reported_against_its_nearest_variant() -> None:
    payload = _payload(PIN_92CCB0F2)
    guard = _step(payload, "validate", "Reject manual RuleSpec changes")
    guard["run"] += "echo ci-pin-property\n"
    admission = _admit(payload)
    report = render_report(admission, review_context(), label="edited")

    assert _kinds(admission) == {"unreviewed"}
    assert [problem.unit.label for problem in admission.problems] == [
        "validate / Reject manual RuleSpec changes"
    ]
    assert "nearest reviewed variant: this step, reviewed in 92ccb0f2" in report
    assert "+  echo ci-pin-property" in report
    assert "NOT admitted (1 problem(s))" in report


def test_a_new_step_is_unreviewed_and_breaks_the_template_order() -> None:
    payload = _payload(PIN_92CCB0F2)
    payload["jobs"]["validate"]["steps"].insert(
        5, {"name": "Brand new step", "shell": "bash", "run": "true\n"}
    )
    admission = _admit(payload)

    assert _kinds(admission) == {"unreviewed", "shape"}
    assert "+validate / Brand new step" in render_report(
        admission, review_context(), label="edited"
    )


def test_reordered_steps_match_no_template() -> None:
    payload = _payload(PIN_92CCB0F2)
    steps = payload["jobs"]["validate"]["steps"]
    first = next(
        i for i, step in enumerate(steps) if step["name"] == "Validate RuleSpec YAML"
    )
    second = next(
        i
        for i, step in enumerate(steps)
        if step["name"] == "Execute RuleSpec companion tests"
    )
    steps[first], steps[second] = steps[second], steps[first]

    assert _kinds(_admit(payload)) == {"shape"}


def test_moving_a_step_to_another_job_is_unreviewed() -> None:
    payload = _payload(PIN_92CCB0F2)
    moved = next(
        step
        for step in payload["jobs"]["validate"]["steps"]
        if step["name"] == "Free runner disk space"
    )
    payload["jobs"]["validate"]["steps"].remove(moved)
    payload["jobs"]["shards"]["steps"].append(moved)

    assert {"unreviewed", "shape"} <= _kinds(_admit(payload))


def test_a_partial_feature_group_matches_no_template() -> None:
    payload = _payload(PIN_92CCB0F2)
    payload["jobs"]["validate"]["steps"] = [
        step
        for step in payload["jobs"]["validate"]["steps"]
        if step["name"] != "Install uv for composer runtime"
    ]

    assert _kinds(_admit(payload)) == {"shape"}


def test_removing_a_whole_group_its_variants_rely_on_is_a_feature_conflict() -> None:
    payload = _payload(PIN_92CCB0F2)
    composer = {name for _, name in FEATURE_STEPS["composer_runtime"]}
    payload["jobs"]["validate"]["steps"] = [
        step
        for step in payload["jobs"]["validate"]["steps"]
        if step["name"] not in composer
    ]
    admission = _admit(payload)

    assert admission.template == PIN_92CCB0F2
    assert admission.removed_groups == ("composer_runtime",)
    assert _kinds(admission) == {"features"}
    assert any(
        "composer_runtime is off" in problem.message for problem in admission.problems
    )


def test_variants_from_pins_with_other_features_conflict() -> None:
    payload = _payload(PIN_92CCB0F2)
    steps = payload["jobs"]["validate"]["steps"]
    index = next(
        i for i, step in enumerate(steps) if step["name"] == "Checkout axiom-encode"
    )
    steps[index] = _step(_payload(PIN_6F11BE26), "validate", "Checkout axiom-encode")
    admission = _admit(payload)

    assert _kinds(admission) == {"features"}
    assert any(
        "isolated_dependencies" in problem.message
        and "validate / Checkout axiom-encode" in problem.message
        for problem in admission.problems
    )


def test_producer_and_consumer_variants_must_have_been_reviewed_together() -> None:
    payload = _payload(PIN_92CCB0F2)
    old_shards = _payload(PIN_0EFFA6A5)["jobs"]["shards"]
    payload["jobs"]["shards"] = {
        **{key: value for key, value in old_shards.items() if key != "steps"},
        "steps": payload["jobs"]["shards"]["steps"],
    }
    admission = _admit(payload)

    assert _kinds(admission) == {"dependency"}
    assert all("needs.shards" in problem.message for problem in admission.problems)


def test_a_missing_input_ci_reads_is_refused() -> None:
    payload = _payload(PIN_92CCB0F2)
    del payload[True]["workflow_call"]["inputs"]["allow-retired-schema-prefreeze"]
    admission = _admit(payload)

    assert "inputs" in _kinds(admission)


def test_duplicate_steps_are_refused() -> None:
    payload = _payload(PIN_92CCB0F2)
    payload["jobs"]["validate"]["steps"].append(
        copy.deepcopy(_step(payload, "validate", "Run repository tests"))
    )

    assert {"duplicate", "shape"} <= _kinds(_admit(payload))


@pytest.mark.parametrize(
    ("job", "name", "edit", "message"),
    [
        (
            *PRECHECK,
            lambda run: run.replace("import ", 'x = "${{ github.sha }}"\nimport ', 1),
            "expression",
        ),
        (
            "shards",
            "Reject unsupported tracked paths",
            lambda run: run.replace("import ", "x = 1  # it's\nimport ", 1),
            "single quote",
        ),
        (
            "validate",
            "Enforce validation waiver ratchet",
            lambda run: run + "cat <<EOF\nunquoted $HOME\nEOF\n",
            "exactly one heredoc",
        ),
        (
            "validate",
            "Enforce validation waiver ratchet",
            lambda run: run.replace("<<'PY'", "<<PY", 1),
            "quote its heredoc",
        ),
        (
            *PRECHECK,
            lambda run: run.replace("import ", "def broken(:\nimport ", 1),
            "does not parse",
        ),
    ],
)
def test_programs_ci_and_bash_could_read_differently_are_refused(
    job: str, name: str, edit: Any, message: str
) -> None:
    payload = _payload(PIN_92CCB0F2)
    step = _step(payload, job, name)
    step["run"] = edit(step["run"])
    admission = _admit(payload)

    assert _kinds(admission) == {"malformed"}
    assert message in admission.problems[0].message


def test_an_unnamed_step_is_refused() -> None:
    payload = _payload(PIN_92CCB0F2)
    payload["jobs"]["validate"]["steps"].append({"uses": "actions/checkout@v4"})

    assert _kinds(_admit(payload)) == {"malformed"}


@pytest.mark.parametrize(
    "expression", ["${{ toJSON(inputs) }}", "${{ inputs['python-version'] }}"]
)
def test_inputs_must_be_read_by_name(expression: str) -> None:
    payload = _payload(PIN_92CCB0F2)
    _step(payload, "validate", "Free runner disk space").setdefault("env", {})["X"] = (
        expression
    )

    assert _kinds(_admit(payload)) == {"malformed"}


# ---------------------------------------------------------------------------
# 6. ci-pin add
# ---------------------------------------------------------------------------


def _git(path: Path, *args: str, check: bool = True) -> str:
    environment = {
        name: value for name, value in os.environ.items() if not name.startswith("GIT_")
    }
    environment.update(GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_NOSYSTEM="1")
    return subprocess.run(
        ["git", "-C", str(path), *args],
        check=check,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=environment,
    ).stdout.strip()


def _commit_workflow(repo: Path, data: bytes, message: str) -> str:
    path = repo / WORKFLOW_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        path.unlink()
    path.write_bytes(data)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", message)
    return _git(repo, "rev-parse", "HEAD")


@pytest.fixture
def github(tmp_path: Path) -> dict[str, Any]:
    """A .github clone whose main has 92ccb0f2's then ff774c29's workflow."""

    repo = tmp_path / "github"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")
    _git(repo, "config", "commit.gpgsign", "false")
    _git(repo, "remote", "add", "origin", GITHUB_URL)
    first = _commit_workflow(repo, packaged_bytes(PIN_92CCB0F2), "92ccb0f2's workflow")
    second = _commit_workflow(repo, packaged_bytes(PIN_FF774C29), "ff774c29's workflow")
    _git(repo, "update-ref", "refs/remotes/origin/main", second)
    destination = tmp_path / "workflows"
    shutil.copytree(WORKFLOW_DIRECTORY, destination)
    manifest = load_manifest(destination)
    (destination / manifest.pop(PIN_FF774C29).file).unlink()
    ci_parity_pins.write_manifest(manifest, destination)
    return {"repo": repo, "first": first, "second": second, "destination": destination}


def _add(github: dict[str, Any], sha: str, **options: Any) -> tuple[int, str]:
    out = io.StringIO()
    code = add_pin(
        sha,
        github["repo"],
        destination=github["destination"],
        main_url=str(github["repo"]),
        out=out,
        **options,
    )
    return code, out.getvalue()


def _snapshot(directory: Path) -> dict[str, bytes]:
    return {path.name: path.read_bytes() for path in sorted(directory.iterdir())}


def test_add_packages_an_admitted_pin_byte_exact(github: dict[str, Any]) -> None:
    code, output = _add(github, github["second"])
    manifest = load_manifest(github["destination"])
    entry = manifest[github["second"]]

    assert code == 0, output
    assert "admitted; every glue unit is reviewed" in output
    assert "template 92ccb0f2, removed feature groups: none" in output
    assert entry.file == f"validate-rulespec-{github['second'][:8]}.yml"
    data = (github["destination"] / entry.file).read_bytes()
    assert data == packaged_bytes(PIN_FF774C29)
    assert (
        entry.blob
        == git_blob_id(data)
        == _git(github["repo"], "rev-parse", f"{github['second']}:{WORKFLOW_PATH}")
    )
    assert (
        _git(github["repo"], "rev-parse", ci_parity_pins.FETCHED_MAIN_REF)
        == github["second"]
    )


def test_add_is_idempotent_and_accepts_an_abbreviation(github: dict[str, Any]) -> None:
    assert _add(github, github["second"])[0] == 0
    before = _snapshot(github["destination"])
    code, output = _add(github, github["second"][:10].upper())

    assert code == 0
    assert "already packaged" in output
    assert _snapshot(github["destination"]) == before


def test_add_refuses_unreviewed_glue_and_writes_nothing(github: dict[str, Any]) -> None:
    payload = _payload(PIN_FF774C29)
    data = packaged_bytes(PIN_FF774C29).replace(
        b"          /opt/axiom-verification/axiom-encode-signing-supervisor \\\n",
        b"          echo ci-pin-property\n          /opt/axiom-verification/axiom-encode-signing-supervisor \\\n",
        1,
    )
    assert data != packaged_bytes(PIN_FF774C29) and payload
    edited = _commit_workflow(github["repo"], data, "edited glue")
    before = _snapshot(github["destination"])
    code, output = _add(github, edited)

    assert code == 1
    assert "NOT admitted" in output
    assert "+  echo ci-pin-property" in output
    assert "Nothing was written" in output
    assert _snapshot(github["destination"]) == before


def test_dry_run_writes_nothing(github: dict[str, Any]) -> None:
    before = _snapshot(github["destination"])
    code, output = _add(github, github["second"], dry_run=True)

    assert code == 0
    assert "dry run: nothing written" in output
    assert _snapshot(github["destination"]) == before


def test_add_refuses_a_commit_that_is_not_on_main(github: dict[str, Any]) -> None:
    _git(github["repo"], "checkout", "-q", "-b", "side", github["first"])
    side = _commit_workflow(github["repo"], packaged_bytes(PIN_FF774C29), "side branch")

    with pytest.raises(PinSourceError, match="not an ancestor of main fetched"):
        _add(github, side)


def test_no_fetch_trusts_the_checkouts_origin_main(github: dict[str, Any]) -> None:
    _git(github["repo"], "update-ref", "refs/remotes/origin/main", github["first"])

    with pytest.raises(PinSourceError, match="the checkout's origin/main"):
        _add(github, github["second"], fetch=False)
    _git(github["repo"], "update-ref", "refs/remotes/origin/main", github["second"])
    assert _add(github, github["second"], fetch=False)[0] == 0


@pytest.mark.parametrize(
    "url",
    [
        "https://github.com/someone-else/.github.git",
        "https://example.com/TheAxiomFoundation/.github.git",
        "https://github.com/TheAxiomFoundation/rulespec-us.git",
    ],
)
def test_add_refuses_another_origin(github: dict[str, Any], url: str) -> None:
    _git(github["repo"], "remote", "set-url", "origin", url)

    with pytest.raises(
        PinSourceError, match="not github.com/TheAxiomFoundation/.github"
    ):
        _add(github, github["second"])


@pytest.mark.parametrize(
    "url",
    [
        "git@github.com:TheAxiomFoundation/.github.git",
        "https://github.com/theaxiomfoundation/.GITHUB",
    ],
)
def test_add_accepts_the_canonical_origin_spelled_otherwise(
    github: dict[str, Any], url: str
) -> None:
    _git(github["repo"], "remote", "set-url", "origin", url)

    assert _add(github, github["second"])[0] == 0


def test_add_refuses_url_rewrites_grafts_and_shallow_clones(
    github: dict[str, Any], tmp_path: Path
) -> None:
    repo = github["repo"]
    _git(repo, "config", "url.https://example.com/.insteadOf", "https://github.com/")
    with pytest.raises(PinSourceError, match="rewrites remote URLs"):
        _add(github, github["second"])
    _git(repo, "config", "--unset", "url.https://example.com/.insteadOf")

    grafts = repo / ".git" / "info" / "grafts"
    grafts.parent.mkdir(exist_ok=True)
    grafts.write_text(f"{github['second']}\n")
    with pytest.raises(PinSourceError, match="grafts"):
        _add(github, github["second"])
    grafts.unlink()

    shallow = tmp_path / "shallow"
    subprocess.run(
        ["git", "clone", "-q", "--depth", "1", f"file://{repo}", str(shallow)],
        check=True,
        env={**os.environ, "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": os.devnull},
    )
    _git(shallow, "remote", "set-url", "origin", GITHUB_URL)
    with pytest.raises(PinSourceError, match="shallow"):
        add_pin(
            github["second"],
            shallow,
            destination=github["destination"],
            main_url=str(repo),
            out=io.StringIO(),
        )


def test_add_ignores_replace_refs(github: dict[str, Any]) -> None:
    # A replace ref swapping the packaged commit for an edited one is ignored.
    edited = _commit_workflow(github["repo"], b"jobs: {}\n", "edited")
    _git(github["repo"], "replace", github["second"], edited)
    _git(github["repo"], "reset", "-q", "--hard", github["second"])

    code, output = _add(github, github["second"])
    assert code == 0, output
    assert (
        github["destination"] / f"validate-rulespec-{github['second'][:8]}.yml"
    ).read_bytes() == (packaged_bytes(PIN_FF774C29))


def test_add_refuses_a_symlinked_workflow(github: dict[str, Any]) -> None:
    repo = github["repo"]
    path = repo / WORKFLOW_PATH
    (repo / "elsewhere.yml").write_bytes(packaged_bytes(PIN_FF774C29))
    path.unlink()
    path.symlink_to("../../elsewhere.yml")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "symlink")
    linked = _git(repo, "rev-parse", "HEAD")

    with pytest.raises(PinSourceError, match="not a regular file"):
        _add(github, linked)


@pytest.mark.parametrize("sha", ["not-a-sha", "abc", "g" * 40])
def test_add_refuses_what_is_not_a_commit_id(github: dict[str, Any], sha: str) -> None:
    with pytest.raises(PinSourceError, match="not a hexadecimal commit id"):
        _add(github, sha)


def test_add_refuses_an_unknown_commit(github: dict[str, Any]) -> None:
    with pytest.raises(PinSourceError, match="names no commit"):
        _add(github, "1" * 40)


def test_cli_lists_every_packaged_pin_with_how_it_was_admitted() -> None:
    out = io.StringIO()

    assert run_ci_pin(Namespace(ci_pin_action="list"), out) == 0
    lines = out.getvalue().splitlines()
    assert len(lines) == len(SUPPORTED_WORKFLOW_PINS)
    by_sha = {line.split()[0]: line for line in lines}
    assert "legacy" in by_sha["615c1df9b9ace7deea84da65efd137f46f8bad2b"]
    assert "template" in by_sha[PIN_92CCB0F2]
    assert "like 92ccb0f2;" in by_sha[PIN_FF774C29]
    assert "like 92ccb0f2 without program_spec_precheck" in by_sha[PIN_DF2DFB53]


def test_cli_reports_a_bad_source_without_a_traceback(github: dict[str, Any]) -> None:
    out = io.StringIO()
    code = run_ci_pin(
        Namespace(
            ci_pin_action="add",
            sha="1" * 40,
            github_checkout=github["repo"],
            no_fetch=True,
            dry_run=True,
        ),
        out,
    )

    assert code == 2
    assert out.getvalue().startswith("ci-pin add: ")


def test_ci_pin_is_an_axiom_encode_subcommand(monkeypatch: pytest.MonkeyPatch) -> None:
    from axiom_encode import cli

    seen: list[Namespace] = []
    monkeypatch.setattr(
        ci_parity_pins, "run_ci_pin", lambda args: seen.append(args) or 0
    )
    monkeypatch.setattr("sys.argv", ["axiom-encode", "ci-pin", "list"])

    with pytest.raises(SystemExit) as exit_info:
        cli.main()
    assert exit_info.value.code == 0
    assert seen[0].ci_pin_action == "list"


def test_marker_and_programs_cannot_collide() -> None:
    # The marker is not Python, so no program can be mistaken for it.
    with pytest.raises(SyntaxError):
        ast.parse(PROGRAM_MARKER)
    assert executed_program(
        "python - <<'PY'\nprint(1)\nPY\n", EXECUTED_PROGRAMS[PRECHECK]
    ) == (f"python - <<'PY'\n{PROGRAM_MARKER}\nPY\n", "print(1)\n")


def test_c_and_python_yaml_loaders_give_the_same_units() -> None:
    data = packaged_bytes(PIN_92CCB0F2)
    pure = yaml.load(data, Loader=yaml.SafeLoader)

    assert [unit.digest for unit in glue_units(pure)] == [
        unit.digest for unit in glue_units(packaged_payload(PIN_92CCB0F2))
    ]


@pytest.mark.parametrize(
    "scalar",
    [
        "on",
        "off",
        "yes",
        "No",
        "True",
        "TRUE",
        "0777",
        "0o17",
        "0x1F",
        "1_000",
        "1:30",
        "1e3",
        ".inf",
        "2026-10-13",
        "Null",
    ],
)
def test_scalars_yaml_readers_disagree_on_are_refused(scalar: str) -> None:
    # PyYAML (YAML 1.1) reads `on` as true, 0777 as 511, 1:30 as 90 and a date
    # as a date; a YAML 1.2 reader reads 0o17 and 1e3 as numbers. A digest must
    # not equate `on` with `true`.
    text = (
        packaged_bytes(PIN_92CCB0F2)
        .decode()
        .replace(
            "fetch-depth: 0\n",
            f"fetch-depth: 0\n          ci-pin-property: {scalar}\n",
            1,
        )
    )
    assert f"ci-pin-property: {scalar}" in text

    with pytest.raises(GlueError, match="read differently by different YAML readers"):
        ci_parity_pins.load_workflow(text.encode())
    quoted = text.replace(f"ci-pin-property: {scalar}", f"ci-pin-property: '{scalar}'")
    assert ci_parity_pins.load_workflow(quoted.encode())


@pytest.mark.parametrize(
    ("addition", "message"),
    [
        ("          anchored: &a value\n", "anchor or alias"),
        ("          tagged: !!str 12\n", "explicit YAML tag"),
    ],
)
def test_anchors_and_explicit_tags_are_refused(addition: str, message: str) -> None:
    text = (
        packaged_bytes(PIN_92CCB0F2)
        .decode()
        .replace("fetch-depth: 0\n", f"fetch-depth: 0\n{addition}", 1)
    )

    with pytest.raises(GlueError, match=message):
        ci_parity_pins.load_workflow(text.encode())


@pytest.mark.parametrize(
    "sha", sorted(SUPPORTED_WORKFLOW_PINS), ids=lambda sha: sha[:8]
)
def test_packaged_workflows_read_the_same_under_either_yaml_schema(sha: str) -> None:
    data = (WORKFLOW_DIRECTORY / SUPPORTED_WORKFLOW_PINS[sha].fixture).read_bytes()
    payload = ci_parity_pins.load_workflow(data)
    strict = yaml.load(data, Loader=ci_parity._WorkflowYamlLoader)

    # Only the top-level `on` key differs between YAML 1.1 and ci's 1.2 loader.
    assert {
        ("on" if key is True else key): value for key, value in payload.items()
    } == strict


def test_git_reads_ancestry_from_commit_objects_only(github: dict[str, Any]) -> None:
    environment = ci_parity_pins._hermetic_git_environment()
    value = subprocess.run(
        ["git", "-C", str(github["repo"]), "config", "--get", "core.commitGraph"],
        env=environment,
        stdout=subprocess.PIPE,
        text=True,
        check=True,
    ).stdout.strip()

    assert value == "false"
    assert environment["GIT_NO_REPLACE_OBJECTS"] == "1"


def _edited_glue(data: bytes) -> bytes:
    edited = data.replace(
        b"          /opt/axiom-verification/axiom-encode-signing-supervisor \\\n",
        b"          echo ci-pin-template\n"
        b"          /opt/axiom-verification/axiom-encode-signing-supervisor \\\n",
        1,
    )
    assert edited != data
    return edited


def _declare_template(
    monkeypatch: pytest.MonkeyPatch, sha: str, features: dict[str, bool]
) -> None:
    declared = {**TEMPLATE_FEATURES, sha: features}
    monkeypatch.setattr(ci_parity_pins, "TEMPLATE_FEATURES", declared)
    monkeypatch.setattr(ci_parity_pins, "TEMPLATE_PINS", tuple(declared))


def test_template_mode_reviews_a_declared_pins_new_glue(
    github: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    new = _commit_workflow(
        github["repo"], _edited_glue(packaged_bytes(PIN_FF774C29)), "new glue"
    )
    with pytest.raises(PinSourceError, match="not declared in TEMPLATE_FEATURES"):
        _add(github, new, template=True)
    assert _add(github, new)[0] == 1

    _declare_template(monkeypatch, new, dict(TEMPLATE_FEATURES[PIN_92CCB0F2]))
    code, output = _add(github, new, template=True)

    assert code == 0, output
    assert "1 newly reviewed glue unit(s)" in output
    reviewed = load_reviewed_glue(github["destination"])
    added = [entry for entry in reviewed if entry.reviewed_in == new[:8]]
    assert [(entry.kind, entry.job, entry.step) for entry in added] == [
        ("step", "validate", "Enforce validation waiver ratchet")
    ]
    # A later pin with the same glue and another pre-check program is admitted.
    later_data = _edited_glue(packaged_bytes(PIN_FF774C29)).replace(
        b"          import json\n", b"          import json\n          # reworded\n", 1
    )
    assert later_data.index(b"# reworded") > later_data.index(b"Reject unmanifested")
    later = _commit_workflow(github["repo"], later_data, "later")
    code, output = _add(github, later)
    assert code == 0, output
    assert "admitted; every glue unit is reviewed" in output


def test_template_mode_refuses_features_its_glue_contradicts(
    github: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    new = _commit_workflow(
        github["repo"], _edited_glue(packaged_bytes(PIN_FF774C29)), "new glue"
    )
    _declare_template(
        monkeypatch,
        new,
        {**TEMPLATE_FEATURES[PIN_92CCB0F2], "exact_guard_base_ref": False},
    )
    before = _snapshot(github["destination"])
    code, output = _add(github, new, template=True)

    assert code == 1
    assert "exact_guard_base_ref" in output
    assert _snapshot(github["destination"]) == before


def test_dependencies_follow_exports_into_python_and_the_path() -> None:
    units = glue_units(packaged_payload(PIN_92CCB0F2))
    pairs, problems = ci_parity_pins._dependencies(units)
    found = {
        (consumer.step, producer.step, reference)
        for consumer, producer, reference in pairs
    }

    assert problems == []
    # A Python reader (os.environ[...]) of a GITHUB_ENV export, not only bash.
    assert (
        "Install changed-file oracle coverage classifier",
        "Reserve isolated dependency directories",
        "GITHUB_ENV AXIOM_DEPENDENCY_ROOT",
    ) in found
    # Every later step of the job runs with a GITHUB_PATH writer's PATH.
    later = {
        consumer
        for consumer, producer, reference in found
        if (producer, reference) == ("Expose axiom-rules-engine on PATH", "GITHUB_PATH")
    }
    names = [
        unit.step for unit in units if unit.kind == "step" and unit.job == "validate"
    ]
    assert later == set(names[names.index("Expose axiom-rules-engine on PATH") + 1 :])
    # Step and job outputs.
    assert ("Run repository tests", None, "needs.shards") in found
    assert any(reference == "steps.toolchain" for _, _, reference in found)


def test_a_path_writer_variant_must_have_been_reviewed_with_its_readers() -> None:
    # 0effa6a5's "Expose axiom-rules-engine on PATH" writes another directory
    # to GITHUB_PATH; every later 92ccb0f2 step ran with 92ccb0f2's.
    payload = _payload(PIN_92CCB0F2)
    steps = payload["jobs"]["validate"]["steps"]
    index = next(
        i
        for i, step in enumerate(steps)
        if step["name"] == "Expose axiom-rules-engine on PATH"
    )
    steps[index] = _step(
        _payload(PIN_0EFFA6A5), "validate", "Expose axiom-rules-engine on PATH"
    )
    admission = _admit(payload)

    assert {"features", "dependency"} <= _kinds(admission)
    assert any("GITHUB_PATH" in problem.message for problem in admission.problems)


def test_template_mode_checks_an_already_packaged_pin(
    github: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    assert _add(github, github["second"])[0] == 0
    before = _snapshot(github["destination"])

    # Promoting it with features its glue contradicts is refused, not "already
    # packaged".
    _declare_template(
        monkeypatch,
        github["second"],
        {**TEMPLATE_FEATURES[PIN_92CCB0F2], "isolated_dependencies": False},
    )
    code, output = _add(github, github["second"], template=True)
    assert code == 1, output
    assert "already packaged" in output and "NOT admitted" in output
    assert _snapshot(github["destination"]) == before

    _declare_template(
        monkeypatch, github["second"], dict(TEMPLATE_FEATURES[PIN_92CCB0F2])
    )
    code, output = _add(github, github["second"], template=True)
    assert code == 0, output
    assert "wrote reviewed-glue.json" in output
    after = _snapshot(github["destination"])
    assert set(after) == set(before)
    assert {
        name: data for name, data in after.items() if name != REVIEWED_GLUE_NAME
    } == {name: data for name, data in before.items() if name != REVIEWED_GLUE_NAME}
    # Its pre-check program is not 92ccb0f2's, but its glue is: nothing new.
    assert after[REVIEWED_GLUE_NAME] == before[REVIEWED_GLUE_NAME]


def test_add_leaves_no_temporary_files(github: dict[str, Any]) -> None:
    assert _add(github, github["second"])[0] == 0

    assert not [
        path.name
        for path in github["destination"].iterdir()
        if path.name.startswith(".")
    ]


@pytest.mark.parametrize(
    "loader", [yaml.SafeLoader, getattr(yaml, "CSafeLoader", None)]
)
def test_portable_yaml_check_is_the_same_under_either_pyyaml_parser(
    monkeypatch: pytest.MonkeyPatch, loader: Any
) -> None:
    # libyaml marks a plain scalar's style "" and the Python parser None.
    if loader is None:
        pytest.skip("PyYAML was built without libyaml")
    monkeypatch.setattr(ci_parity_pins, "_YAML_LOADER", loader)
    text = packaged_bytes(PIN_92CCB0F2).decode()

    payload = ci_parity_pins.load_workflow(text.encode())
    assert [unit.digest for unit in glue_units(payload)] == [
        unit.digest for unit in glue_units(packaged_payload(PIN_92CCB0F2))
    ]
    for addition, message in (
        ("          ci-pin-property: on\n", "read differently"),
        ("          ci-pin-property: 1e3\n", "read differently"),
        ("          anchored: &a value\n", "anchor or alias"),
        ("          tagged: !!str 12\n", "explicit YAML tag"),
    ):
        edited = text.replace("fetch-depth: 0\n", f"fetch-depth: 0\n{addition}", 1)
        with pytest.raises(GlueError, match=message):
            ci_parity_pins.load_workflow(edited.encode())
