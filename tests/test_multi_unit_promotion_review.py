"""A multi-unit promotion with the generalist review and the source pin.

The batch form (#2301) meets ``promote-reviewed-candidate --require-review``
(#2318) and the candidate source pin (#2323):

- with ``--require-review``, every unit is reviewed on its own signed bytes,
  in unit order, after every unit is validated and before the first
  signature; the first unit without a passed review stops the batch;
- without it, no reviewer runs;
- ``--review-record-file`` holds one record, so a batch of several units
  refuses it before validation (issue #2319 tracks the list form);
- the admission and the pre-approval script check the pin of every unit.

No test calls a model: the reviewer CLI is replaced by scripted text.
"""

from __future__ import annotations

import base64
import json
import subprocess
import sys
from pathlib import Path

import pytest

import scripts.check_candidate_source_pin as pin_mod
from axiom_encode import reviewed_pr_admission as admission
from axiom_encode.generalist_review_chunks import (
    CHUNK_SOURCE_CHARS,
    MAX_CHUNKS_PER_FILE,
)
from axiom_encode.harness import evals as evals_module
from axiom_encode.harness import validator_pipeline
from axiom_encode.reviewed_pr_admission import AdmissionError, verify_reviewed_pr
from tests.test_multi_unit_promotion_admission import (
    HEAD,
    ROOT,
    Reads,
    batch,
    make_api,
    unit,
)
from tests.test_multi_unit_promotion_cli import (
    NAMES,
    Run,
    batch_admission,
    candidate,
    citation,
    manifest_of,
    primary,
)

PASS = '{"score": 9.0, "passed": true, "blocking_issues": []}'
FAIL = '{"score": 3.0, "passed": false, "blocking_issues": ["a defect"]}'
REVIEW_LINE = "reviewed candidate generalist review: "


@pytest.fixture
def on(monkeypatch):
    monkeypatch.setattr(admission, "MULTI_UNIT_PROMOTION_ENABLED", True)


def _unit_of(text: str) -> str:
    """The unit whose source text a prompt or a review source carries."""
    found = [name for name in NAMES if f"source text of {citation(name)}" in text]
    assert len(found) == 1, text[:200]
    return found[0]


class Reviewed(Run):
    """A batch run with the reviewer scripted per unit and every step logged."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.events: list[tuple[str, str]] = []
        self.staged: dict[str, tuple[bytes, bytes]] = {}
        self.cwds: dict[str, Path] = {}
        self.outputs: dict[str, object] = {}
        self.raise_for: str | None = None
        self.truncate_for: str | None = None
        real_sign = self._sign

        def sign(payload, broker):
            self.events.append(("sign", payload["citation"].rsplit("/", 1)[1]))
            return real_sign(payload, broker)

        self._sign = sign

    def __call__(self, monkeypatch):
        real_review = validator_pipeline.run_generalist_review
        real_source = evals_module.generalist_review_source

        def review(rulespec_file, **kwargs):
            name = Path(rulespec_file).stem
            self.events.append(("review", name))
            self.cwds[name] = Path(kwargs["cwd"])
            self.staged[name] = (
                Path(rulespec_file).read_bytes(),
                Path(rulespec_file).with_name(f"{name}.test.yaml").read_bytes(),
            )
            if name == self.raise_for:
                raise RuntimeError("reviewer process crashed")
            return real_review(rulespec_file, **kwargs)

        def source(text, **kwargs):
            if self.truncate_for is not None and _unit_of(text) == self.truncate_for:
                text = text + "word\n" * (MAX_CHUNKS_PER_FILE * CHUNK_SOURCE_CHARS)
            return real_source(text, **kwargs)

        def reviewer(prompt, **_kwargs):
            output = self.outputs.get(_unit_of(prompt), PASS)
            return output if isinstance(output, tuple) else (output, 0)

        monkeypatch.setattr(validator_pipeline, "run_generalist_review", review)
        monkeypatch.setattr(evals_module, "generalist_review_source", source)
        monkeypatch.setattr(validator_pipeline, "run_claude_code", reviewer)
        super().__call__(monkeypatch)

    def reviews(self) -> list[str]:
        return [name for kind, name in self.events if kind == "review"]


@pytest.fixture
def reviewed(tmp_path, on):
    repo, head = candidate(tmp_path)
    return Reviewed(tmp_path, repo, head, batch_admission(repo, head))


def _review_records(output: str) -> list[dict]:
    return [
        json.loads(line[len(REVIEW_LINE) :])
        for line in output.splitlines()
        if line.startswith(REVIEW_LINE)
    ]


# --- (a) every review passes: every unit is signed ---------------------------


def test_every_unit_is_reviewed_on_its_own_bytes_before_any_signature(
    reviewed, monkeypatch, capsys
):
    reviewed.args.require_review = True
    reviewed(monkeypatch)

    # One review per unit, in unit order, all after validation and before the
    # first signature.
    assert reviewed.validated == list(NAMES)
    assert reviewed.events == [("review", name) for name in NAMES] + [
        ("sign", name) for name in NAMES
    ]
    for name in NAMES:
        rule = reviewed.repo / primary(name)
        assert reviewed.staged[name] == (
            rule.read_bytes(),
            rule.with_name(f"{name}.test.yaml").read_bytes(),
        )
        assert manifest_of(reviewed.repo, name).exists()
        # The reviewer runs in the content root of the unit, as for one unit.
        assert reviewed.cwds[name] == reviewed.repo / primary(name).split("/")[0]
    records = _review_records(capsys.readouterr().out)
    assert [record["applied"]["rulespec_path"] for record in records] == [
        primary(name) for name in NAMES
    ]
    assert all(record["status"] == "passed" for record in records)
    # Each record binds its unit's signed bytes.
    for name, record in zip(NAMES, records):
        signed = json.loads(manifest_of(reviewed.repo, name).read_text())
        assert sorted(
            (entry["path"], entry["sha256"]) for entry in signed["applied_files"]
        ) == sorted((entry["path"], entry["sha256"]) for entry in record["files"])
    assert reviewed.revalidations == 2


# --- (b) the second unit's review does not pass: nothing is signed ------------


@pytest.mark.parametrize("mode", ["failed", "unavailable", "partial", "raises"])
def test_the_second_unit_without_a_passed_review_stops_the_batch(
    reviewed, monkeypatch, capsys, mode
):
    reviewed.args.require_review = True
    if mode == "failed":
        reviewed.outputs["b"] = FAIL
    elif mode == "unavailable":
        reviewed.outputs["b"] = ("reviewer_unavailable: no reviewer CLI", 1)
    elif mode == "partial":
        reviewed.truncate_for = "b"
    else:
        reviewed.raise_for = "b"

    with pytest.raises(RuntimeError) as refused:
        reviewed(monkeypatch)

    message = str(refused.value)
    assert message.startswith(f"unit {citation('b')} ({primary('b')}): ")
    if mode == "raises":
        assert message.endswith("reviewer process crashed")
    else:
        assert f"has none: status {mode}" in message
        assert message.endswith("Nothing was signed.")
    # Every unit was validated; the review stopped at the second unit; no
    # signature was requested for any unit and no manifest exists.
    assert reviewed.validated == list(NAMES)
    assert reviewed.reviews() == ["a", "b"]
    assert reviewed.signed == []
    assert reviewed.installs == []
    assert reviewed.revalidations == 1
    reviewed.assert_nothing_written()


# --- (c) the switch off: no reviewer ------------------------------------------


@pytest.mark.parametrize("value", ["absent", False, "true", 1])
def test_without_the_flag_no_unit_is_reviewed(tmp_path, on, monkeypatch, capsys, value):
    repo, head = candidate(tmp_path)
    run = Reviewed(tmp_path, repo, head, batch_admission(repo, head))
    if value != "absent":
        run.args.require_review = value
    run(monkeypatch)

    assert run.reviews() == []
    assert run.events == [("sign", name) for name in NAMES]
    assert capsys.readouterr().out.splitlines() == [
        "promoted reviewed candidate "
        f".axiom/encoding-manifests/us/statutes/7/2015/{name}.json"
        for name in NAMES
    ]


# --- (d) the record file holds one record ------------------------------------


def test_a_review_record_file_with_a_batch_is_refused_before_validation(
    reviewed, monkeypatch, tmp_path
):
    record_dir = tmp_path / "records"
    record_dir.mkdir()
    reviewed.args.require_review = True
    reviewed.args.review_record_file = str(record_dir / "review.json")
    reviewed.args.run_attempt = "1"

    with pytest.raises(ValueError) as refused:
        reviewed(monkeypatch)

    assert "the review record file holds one record" in str(refused.value)
    assert "a batch needs the list form, see issue #2319" in str(refused.value)
    assert "3 units" in str(refused.value)
    assert reviewed.validated == []
    assert reviewed.revalidations == 0
    assert reviewed.events == []
    assert not (record_dir / "review.json").exists()
    reviewed.assert_nothing_written()


def test_a_review_record_file_with_a_batch_of_one_writes_its_one_record(
    tmp_path, on, monkeypatch
):
    repo, head = candidate(tmp_path)
    run = Reviewed(tmp_path, repo, head, batch_admission(repo, head, ("a",)))
    record_dir = tmp_path / "records"
    record_dir.mkdir()
    run.args.require_review = True
    run.args.review_record_file = str(record_dir / "review.json")
    run.args.run_attempt = "2"

    run(monkeypatch)

    # One unit: the record file of a single-unit promotion, unchanged.
    written = json.loads((record_dir / "review.json").read_text())
    assert written["schema"] == "axiom-encode/promotion-generalist-review/v1"
    assert written["citation"] == citation("a")
    assert written["rulespec_path"] == primary("a")
    assert written["run_attempt"] == 2
    assert written["review_passed"] is True
    assert run.events == [("review", "a"), ("sign", "a")]


# --- (e) the source pin of every unit -----------------------------------------


UNITS3 = [unit(1), unit(2), unit(3)]
UNBOUND = b"format: rulespec/v1\nmodule:\n  summary: no pin\nrules: []\n"


def _replace_primary(api: dict, path: str, raw: bytes) -> None:
    entry = api[f"{ROOT}/contents/{path}?ref={HEAD}"]
    entry["content"] = base64.b64encode(raw).decode()
    entry["size"] = len(raw)


@pytest.mark.parametrize("index", [0, 1, 2])
def test_admission_refuses_a_batch_whose_unit_lacks_the_pin_and_names_it(on, index):
    api = make_api(UNITS3)
    target = UNITS3[index]
    _replace_primary(api, target[1], UNBOUND)
    with pytest.raises(AdmissionError) as refused:
        verify_reviewed_pr(batch(UNITS3), fetch=Reads(api))
    message = str(refused.value)
    assert message.startswith(f"unit {target[0]} ({target[1]}): ")
    assert "candidate module is not bound to its source" in message
    assert "bind-source-verification" in message
    assert "`module.source_verification`" in refused.value.detail


def test_admission_refuses_a_unit_pinned_to_another_units_citation(on):
    api = make_api(UNITS3)
    other = make_api([UNITS3[0]])[f"{ROOT}/contents/{UNITS3[0][1]}?ref={HEAD}"]
    _replace_primary(api, UNITS3[1][1], base64.b64decode(other["content"]))
    with pytest.raises(AdmissionError) as refused:
        verify_reviewed_pr(batch(UNITS3), fetch=Reads(api))
    assert str(refused.value).startswith(f"unit {UNITS3[1][0]} ({UNITS3[1][1]}): ")
    assert "not the promotion citation" in str(refused.value)


def test_admission_admits_a_batch_whose_every_unit_carries_its_pin(on):
    result = verify_reviewed_pr(batch(UNITS3), fetch=Reads(make_api(UNITS3)))
    assert [record["binding"]["citation"] for record in result["records"]] == [
        target[0] for target in UNITS3
    ]


def _batch_environment(units=UNITS3, lead=0, **overrides):
    wrapper = {
        "schema": "axiom-encode/reviewed-pr-promotion/v1",
        "rulespec_pr": 42,
        "transaction": {},
        "units": [target[0] for target in units],
    }
    return {
        "SOURCE_BUNDLE_JSON": json.dumps(wrapper),
        "COUNTRY": "us",
        "RULESPEC_REF": HEAD,
        "REPLACE_RULESPEC_PATH": units[lead][1],
        "CITATION": units[lead][0],
        "GH_TOKEN": "token-value",
        **overrides,
    }


def _serving(unbound_path: str | None = None):
    api = make_api(UNITS3)
    calls = []

    def fetch(repository, ref, path, token):
        calls.append((repository, ref, path))
        if path == unbound_path:
            return UNBOUND
        return base64.b64decode(api[f"{ROOT}/contents/{path}?ref={HEAD}"]["content"])

    return fetch, calls


def test_the_script_checks_every_unit_of_a_batch(on):
    fetch, calls = _serving()
    status, message = pin_mod.evaluate(
        _batch_environment(lead=1), fetch=fetch, admission=admission
    )
    assert status == 0, message
    assert calls == [
        ("TheAxiomFoundation/rulespec-us", HEAD, target[1]) for target in UNITS3
    ]
    assert "all 3 promotion units" in message


@pytest.mark.parametrize("index", [0, 1, 2])
def test_the_script_refuses_the_batch_whose_unit_lacks_the_pin(on, index):
    target = UNITS3[index]
    fetch, calls = _serving(unbound_path=target[1])
    status, message = pin_mod.evaluate(
        _batch_environment(), fetch=fetch, admission=admission
    )
    assert status == 1
    assert f"unit {target[0]} ({target[1]}): " in message
    assert "candidate module is not bound to its source" in message
    assert "Nothing was approved or signed." in message
    # It stops at the first unit without its pin.
    assert [call[2] for call in calls] == [unit[1] for unit in UNITS3[: index + 1]]


def test_the_script_refuses_a_batch_while_the_switch_is_off():
    assert admission.MULTI_UNIT_PROMOTION_ENABLED is False
    status, message = pin_mod.evaluate(
        _batch_environment(), fetch=pin_mod_forbidden, admission=admission
    )
    assert status == 1
    assert "multi-unit promotion is switched off" in message


@pytest.mark.parametrize(
    "overrides",
    [
        {"CITATION": "us-az/manual/des/faa5/medical/block-09"},
        {"REPLACE_RULESPEC_PATH": "us-az/policies/des/faa5/medical/block-03.yaml"},
    ],
)
def test_the_script_refuses_a_lead_that_is_not_one_of_the_units(on, overrides):
    status, message = pin_mod.evaluate(
        _batch_environment(**overrides), fetch=pin_mod_forbidden, admission=admission
    )
    assert status == 1
    assert "must be one of the promotion units" in message


@pytest.mark.parametrize(
    "units",
    [[], ["a", "a"], [1], "us-az/manual/x", [unit(i)[0] for i in range(1, 8)]],
)
def test_the_script_refuses_a_malformed_unit_list_without_a_read(on, units):
    env = _batch_environment()
    wrapper = json.loads(env["SOURCE_BUNDLE_JSON"])
    wrapper["units"] = units
    env["SOURCE_BUNDLE_JSON"] = json.dumps(wrapper)
    status, message = pin_mod.evaluate(
        env, fetch=pin_mod_forbidden, admission=admission
    )
    assert status == 1
    assert "could not read the candidate primaries" in message


def pin_mod_forbidden(*_args):
    raise AssertionError("a refused batch must not read GitHub")


def test_the_script_maps_a_unit_on_a_bare_interpreter():
    """The unit path resolver loads without the package, as in the job."""
    scripts = Path(pin_mod.__file__).resolve().parent
    program = (
        "import sys; sys.path.insert(0, sys.argv[1]);"
        "import check_candidate_source_pin as m;"
        "print(m.unit_primary_path(sys.argv[2]));"
        "print('axiom_encode' in sys.modules)"
    )
    completed = subprocess.run(
        [sys.executable, "-I", "-c", program, str(scripts), UNITS3[1][0]],
        env={"PATH": ""},
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.splitlines() == [UNITS3[1][1], "False"]


@pytest.mark.parametrize(
    "cited",
    ["us/statute/26/21", "us/regulation/7/273/9", UNITS3[0][0]],
)
def test_the_script_maps_a_unit_as_the_protected_job_does(cited):
    from axiom_encode.prepare_signed_backfill import citation_rulespec_path

    assert pin_mod.unit_primary_path(cited) == str(citation_rulespec_path(cited))
