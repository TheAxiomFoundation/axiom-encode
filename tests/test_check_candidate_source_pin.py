"""The pre-approval refusal of a reviewed candidate without a source pin."""

from __future__ import annotations

import ast
import json
import subprocess
import sys
import urllib.error
from pathlib import Path

import pytest
import yaml

import scripts.check_candidate_source_pin as pin_mod
from axiom_encode import infra_checks

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "check_candidate_source_pin.py"
WORKFLOW = ROOT / ".github" / "workflows" / "targeted-signed-reencode.yml"
STEP = "Refuse a reviewed candidate without a source pin"
CITATION = "xx-yy/manual/unit/block-1"
PATH = "xx-yy/policies/unit/block-1.yaml"
REF = "a" * 40
PIN = "d" * 64
PROMOTION = json.dumps(
    {"schema": "axiom-encode/reviewed-pr-promotion/v1", "rulespec_pr": 7}
)
BOUND = (
    "format: rulespec/v1\nmodule:\n  source_verification:\n"
    f"    corpus_citation_path: {CITATION}\n    source_sha256: {PIN}\nrules: []\n"
).encode()
UNBOUND = BOUND.replace(f"    source_sha256: {PIN}\n".encode(), b"")


def environment(**overrides: str) -> dict[str, str]:
    return {
        "SOURCE_BUNDLE_JSON": PROMOTION,
        "COUNTRY": "xx",
        "RULESPEC_REF": REF,
        "REPLACE_RULESPEC_PATH": PATH,
        "CITATION": CITATION,
        "GH_TOKEN": "token-value",
        **overrides,
    }


def serving(raw: bytes):
    calls = []

    def fetch(repository, ref, path, token):
        calls.append((repository, ref, path, token))
        return raw

    return fetch, calls


def forbidden_fetch(*_args):
    raise AssertionError("a non-promotion dispatch must not read GitHub")


@pytest.mark.parametrize(
    "source_bundle",
    [
        "",
        "[]",
        "not json",
        '["axiom-encode/reviewed-pr-promotion/v1"]',
        '{"schema": "axiom-encode/atomic-source-transaction/v5"}',
        '{"schema": "axiom-encode/reviewed-repair-transaction/v1"}',
    ],
)
def test_every_other_dispatch_passes_without_a_read(source_bundle):
    status, message = pin_mod.evaluate(
        environment(SOURCE_BUNDLE_JSON=source_bundle), fetch=forbidden_fetch
    )
    assert status == 0
    assert "not a reviewed-candidate promotion" in message


def test_a_bound_candidate_passes_and_is_read_at_the_exact_ref():
    fetch, calls = serving(BOUND)
    status, message = pin_mod.evaluate(environment(), fetch=fetch)
    assert status == 0
    assert calls == [("TheAxiomFoundation/rulespec-xx", REF, PATH, "token-value")]
    assert "protected job still binds" in message


def test_an_unbound_candidate_is_refused_with_the_command():
    fetch, _calls = serving(UNBOUND)
    status, message = pin_mod.evaluate(environment(), fetch=fetch)
    assert status == 1
    assert "REFUSED before the signing approval" in message
    assert "source_sha256` is missing" in message
    assert "axiom-encode bind-source-verification" in message
    assert "Nothing was approved or signed" in message


@pytest.mark.parametrize(
    "raw",
    [
        BOUND.replace(CITATION.encode(), b"xx-yy/manual/unit/block-2"),
        BOUND.replace(PIN.encode(), PIN[:-1].encode()),
        BOUND.replace(b"  source_verification:\n", b"  source_verification: &a\n"),
        b"module: {source_verification: {}}\n",
        b"\xff",
    ],
)
def test_a_malformed_pin_or_another_shape_fails_closed(raw):
    fetch, _calls = serving(raw)
    status, message = pin_mod.evaluate(environment(), fetch=fetch)
    assert status == 1
    assert "REFUSED" in message


@pytest.mark.parametrize(
    "overrides",
    [
        {"COUNTRY": "USA"},
        {"COUNTRY": ""},
        {"RULESPEC_REF": "main"},
        {"RULESPEC_REF": REF[:-1]},
        {"RULESPEC_REF": REF.upper()},
        {"REPLACE_RULESPEC_PATH": ""},
        {"REPLACE_RULESPEC_PATH": "xx-yy/policies/../block-1.yaml"},
        {"REPLACE_RULESPEC_PATH": "xx-yy/policies/unit/block-1.test.yaml"},
        {"REPLACE_RULESPEC_PATH": "xx-yy/policies/unit/block-1.json"},
        {"REPLACE_RULESPEC_PATH": "xx-yy/policies/unit/block 1.yaml"},
        {"CITATION": ""},
        {"CITATION": "xx-yy/manual//block-1"},
    ],
)
def test_inputs_that_name_no_exact_candidate_are_refused_without_a_read(overrides):
    status, message = pin_mod.evaluate(environment(**overrides), fetch=forbidden_fetch)
    assert status == 1
    assert "do not name one exact candidate primary" in message


def test_a_failed_read_fails_closed():
    def fetch(*_args):
        raise pin_mod.CandidateReadError("GitHub did not return the file (HTTP 404)")

    status, message = pin_mod.evaluate(environment(), fetch=fetch)
    assert status == 1
    assert "could not read the candidate primary" in message
    assert "HTTP 404" in message


class _Response:
    def __init__(self, raw: bytes, status: int = 200):
        self.raw, self.status = raw, status

    def read(self, limit: int) -> bytes:
        return self.raw[:limit]

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False


def _opener(monkeypatch, responder):
    requests = []

    class Opener:
        def open(self, request, timeout):
            requests.append(request)
            return responder(request)

    monkeypatch.setattr(
        pin_mod.urllib.request, "build_opener", lambda *handlers: Opener()
    )
    return requests


def test_fetch_reads_the_fixed_origin_with_the_token(monkeypatch):
    requests = _opener(monkeypatch, lambda request: _Response(BOUND))
    raw = pin_mod.fetch_candidate_primary(
        "TheAxiomFoundation/rulespec-xx", REF, PATH, "t"
    )
    assert raw == BOUND
    [request] = requests
    assert request.full_url == (
        f"https://api.github.com/repos/TheAxiomFoundation/rulespec-xx/contents/{PATH}"
        f"?ref={REF}"
    )
    assert request.get_header("Authorization") == "Bearer t"
    assert request.get_header("Accept") == "application/vnd.github.raw+json"


def test_fetch_repeats_without_a_token_that_is_not_accepted(monkeypatch):
    def responder(request):
        if request.get_header("Authorization"):
            raise urllib.error.HTTPError(request.full_url, 403, "no", {}, None)
        return _Response(BOUND)

    requests = _opener(monkeypatch, responder)
    assert pin_mod.fetch_candidate_primary("r/x", REF, PATH, "t") == BOUND
    assert [bool(r.get_header("Authorization")) for r in requests] == [True, False]


def test_fetch_without_a_token_asks_once(monkeypatch):
    requests = _opener(monkeypatch, lambda request: _Response(BOUND))
    assert pin_mod.fetch_candidate_primary("r/x", REF, PATH, "") == BOUND
    assert len(requests) == 1
    assert requests[0].get_header("Authorization") is None


@pytest.mark.parametrize(
    "failure",
    [
        lambda request: (_ for _ in ()).throw(
            urllib.error.HTTPError(request.full_url, 404, "missing", {}, None)
        ),
        lambda request: (_ for _ in ()).throw(urllib.error.URLError("down")),
        lambda request: (_ for _ in ()).throw(TimeoutError()),
        lambda request: _Response(b"", status=206),
        lambda request: _Response(b"x" * (pin_mod.MAX_MODULE_BYTES + 1)),
    ],
)
def test_fetch_failures_raise_a_read_error(monkeypatch, failure):
    _opener(monkeypatch, failure)
    with pytest.raises(pin_mod.CandidateReadError):
        pin_mod.fetch_candidate_primary("r/x", REF, PATH, "t")


def test_fetch_refuses_a_redirect():
    with pytest.raises(pin_mod.CandidateReadError):
        pin_mod._NoRedirect().redirect_request(None, None, 302, "", {}, "https://e/")


def test_main_reports_a_refusal_as_an_error_annotation(monkeypatch, capsys):
    monkeypatch.setattr(pin_mod, "evaluate", lambda env: (1, "source-pin: REFUSED x"))
    assert pin_mod.main() == 1
    captured = capsys.readouterr()
    assert "source-pin: REFUSED x" in captured.err
    assert captured.out.startswith("::error title=Candidate has no source pin::")
    monkeypatch.setattr(pin_mod, "evaluate", lambda env: (0, "source-pin: fine"))
    assert pin_mod.main() == 0
    assert capsys.readouterr().out == "source-pin: fine\n"


def test_the_script_runs_isolated_on_a_bare_interpreter():
    """No virtual environment, no package import: `python3 -I` as in the job."""
    completed = subprocess.run(
        [sys.executable, "-I", str(SCRIPT)],
        env={"SOURCE_BUNDLE_JSON": "[]", "PATH": ""},
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "not a reviewed-candidate promotion" in completed.stdout


def test_the_script_and_the_reader_import_only_the_standard_library():
    allowed = set(sys.stdlib_module_names) | {"__future__"}
    for path in (SCRIPT, ROOT / "src" / "axiom_encode" / "reviewed_pr_admission.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                assert node.level == 0, path
                names = [node.module or ""]
            else:
                continue
            assert {name.split(".")[0] for name in names} <= allowed, (path, names)


# --- workflow contract ---


def _job():
    return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))["jobs"][
        "attempt_budget"
    ]


def test_the_step_runs_before_any_approval_in_the_unprotected_job():
    workflow = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    job = workflow["jobs"]["attempt_budget"]
    names = [step.get("name") for step in job["steps"]]
    index = names.index(STEP)
    assert names.index("Checkout axiom-encode") < index
    assert names.index("Refuse an encoder head that no version bump covers") < index
    assert index < names.index("Enforce failed-attempt budget")
    # The job stays unprotected and single-purpose; the encode job waits for it.
    assert "environment" not in job
    assert "permissions" not in job
    assert "continue-on-error" not in job
    assert workflow["permissions"] == {"actions": "read", "contents": "read"}
    assert workflow["jobs"]["encode"]["needs"] == "attempt_budget"
    assert workflow["jobs"]["encode"]["environment"]


def test_the_step_cannot_be_skipped_and_gets_only_dispatch_inputs_and_the_read_token():
    step = next(step for step in _job()["steps"] if step.get("name") == STEP)
    assert "if" not in step
    assert "continue-on-error" not in step
    assert "uses" not in step
    assert step["env"] == {
        "GH_TOKEN": "${{ github.token }}",
        "CITATION": "${{ inputs.citation }}",
        "COUNTRY": "${{ inputs.country }}",
        "RULESPEC_REF": "${{ inputs.rulespec_ref }}",
        "REPLACE_RULESPEC_PATH": "${{ inputs.replace_rulespec_path }}",
        "SOURCE_BUNDLE_JSON": "${{ inputs.source_bundle_json }}",
    }
    assert step["run"].strip() == (
        'python3 -I "$GITHUB_WORKSPACE/axiom-encode/scripts/'
        'check_candidate_source_pin.py"'
    )
    assert "secrets." not in json.dumps(step)


def test_the_step_has_a_registered_check_that_names_the_command():
    check_id = infra_checks.step_check_id(STEP)
    definition = next(
        item for item in infra_checks.STEP_CHECK_DEFINITIONS if item["id"] == check_id
    )
    assert definition["site"].endswith(f":{STEP}")
    assert "bind-source-verification" in definition["fix"]


def test_the_environment_names_match_the_script():
    step = next(step for step in _job()["steps"] if step.get("name") == STEP)
    source = SCRIPT.read_text(encoding="utf-8")
    for name in step["env"]:
        assert f'"{name}"' in source, name
