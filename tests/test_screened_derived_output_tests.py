from pathlib import Path

import pytest
import yaml

from axiom_encode import cli


def fixture(tmp_path):
    root = tmp_path / "rulespec-ca" / "ca"
    relative = Path("policies/example.yaml")
    policy = root / relative
    policy.parent.mkdir(parents=True)
    policy.write_text("""format: rulespec/v1
rules:
  - name: result
    kind: derived
    entity: Person
    period: Year
    dtype: Integer
    versions:
      - effective_from: '2025-01-01'
        formula: '0'
""")
    test = policy.with_suffix(".test.yaml")
    test.write_text(
        "# preserve baseline bytes\n- name: broken_baseline\n  output: {}\n"
    )
    return dict(
        rules_file=policy,
        test_file=test,
        repo_path=root,
        relative_output=relative,
        issues=[
            "Derived rule missing companion output coverage: `ca:policies/example#result`"
        ],
    )


@pytest.mark.parametrize(
    "failure",
    [
        None,
        [{"error": "division by zero"}],
        [{"error": "wrong output"}],
        [{"error": "compile error"}],
    ],
)
def test_rejection_preserves_original(tmp_path, failure):
    args = fixture(tmp_path)
    before = args["test_file"].read_bytes()
    checker = None if failure is None else lambda path: failure
    assert (
        cli._append_generated_derived_output_tests_if_missing(
            **args, test_failure_checker=checker
        )
        == []
    )
    assert args["test_file"].read_bytes() == before


def test_setup_failure_preserves_original(tmp_path):
    args = fixture(tmp_path)
    before = args["test_file"].read_bytes()

    def checker(path):
        raise FileNotFoundError("engine unavailable")

    assert (
        cli._append_generated_derived_output_tests_if_missing(
            **args, test_failure_checker=checker
        )
        == []
    )
    assert args["test_file"].read_bytes() == before


def test_generated_only_screen_cannot_mask_baseline_failure(tmp_path):
    args = fixture(tmp_path)
    before = args["test_file"].read_bytes()
    calls = []

    def checker(path):
        assert args["test_file"].read_bytes() == before
        cases = yaml.safe_load(path.read_text())
        assert [c["name"] for c in cases] == ["auto_output_result"]
        assert cases[0]["output"] == {"ca:policies/example#result": 0}
        calls.append(path)
        return []

    assert cli._append_generated_derived_output_tests_if_missing(
        **args, test_failure_checker=checker
    ) == ["auto_output_result"]
    assert len(calls) == 1
    assert yaml.safe_load(args["test_file"].read_text())[0]["name"] == "broken_baseline"


def test_same_name_not_reused(tmp_path):
    args = fixture(tmp_path)
    args["test_file"].write_text("- name: auto_output_result\n  output: {}\n")
    before = args["test_file"].read_bytes()

    def checker(path):
        pytest.fail("same-name baseline must not become generated evidence")

    assert (
        cli._append_generated_derived_output_tests_if_missing(
            **args, test_failure_checker=checker
        )
        == []
    )
    assert args["test_file"].read_bytes() == before


def test_canonical_overlay_screens_generated_only(tmp_path, monkeypatch):
    args = fixture(tmp_path)
    before = args["test_file"].read_bytes()

    def execute(path, **kwargs):
        assert path.name == args["test_file"].name
        assert kwargs["root"].name == "ca"
        assert kwargs["root"].parent.name == "rulespec-ca"
        assert args["test_file"].read_bytes() == before
        assert len(yaml.safe_load(path.read_text())) == 1
        return []

    monkeypatch.setattr(cli, "_rulespec_companion_test_failures", execute)
    args["policy_repo_path"] = args.pop("repo_path")
    assert cli._append_generated_derived_output_tests_in_overlay(
        **args, axiom_rules_path=tmp_path / "engine"
    ) == ["auto_output_result"]


def test_batch_bound_keeps_unattempted_coverage_missing(tmp_path):
    args = fixture(tmp_path)
    payload = yaml.safe_load(args["rules_file"].read_text())
    rule = payload["rules"][0]
    payload["rules"] = [dict(rule, name=f"result_{i}") for i in range(51)]
    args["rules_file"].write_text(yaml.safe_dump(payload))
    args["issues"] = [
        f"Derived rule missing companion output coverage: `ca:policies/example#result_{i}`"
        for i in range(51)
    ]
    calls = []

    def checker(path):
        calls.append(yaml.safe_load(path.read_text()))
        return []

    added = cli._append_generated_derived_output_tests_if_missing(
        **args, test_failure_checker=checker
    )
    assert len(calls) == 1 and len(calls[0]) == len(added) == 50
    covered = cli._rulespec_test_output_keys(args["test_file"])
    assert len(covered) == 50
    assert (
        len(
            set(cli._missing_derived_output_targets_from_issues(args["issues"]))
            - covered
        )
        == 1
    )


def test_production_overlay_missing_engine_preserves_original(tmp_path, monkeypatch):
    args = fixture(tmp_path)
    before = args["test_file"].read_bytes()

    def fail(*args, **kwargs):
        raise FileNotFoundError("missing engine")

    monkeypatch.setattr(cli, "_rulespec_companion_test_failures", fail)
    args["policy_repo_path"] = args.pop("repo_path")
    assert (
        cli._append_generated_derived_output_tests_in_overlay(
            **args, axiom_rules_path=tmp_path / "missing"
        )
        == []
    )
    assert args["test_file"].read_bytes() == before


@pytest.mark.parametrize("caller", ["cli", "harness"])
@pytest.mark.parametrize("failure", [False, True])
def test_production_callers_execute_shared_screen(
    tmp_path, monkeypatch, caller, failure
):
    from types import SimpleNamespace

    from axiom_encode.harness import evals

    args = fixture(tmp_path)
    before = args["test_file"].read_bytes()
    calls = []

    def execute(path, **kwargs):
        calls.append(kwargs)
        assert args["test_file"].read_bytes() == before
        assert [c["name"] for c in yaml.safe_load(path.read_text())] == [
            "auto_output_result"
        ]
        return [{"message": "division by zero"}] if failure else []

    monkeypatch.setattr(cli, "_rulespec_companion_test_failures", execute)
    if caller == "cli":
        result = cli._try_repair_generated_derived_output_tests_for_apply(
            SimpleNamespace(output_file=str(args["rules_file"])),
            output_root=args["repo_path"],
            policy_repo_path=args["repo_path"],
            axiom_rules_path=tmp_path / "engine",
            issues=args["issues"],
        )
    else:
        result = evals._apply_generated_eval_repairs(
            rulespec_file=args["rules_file"],
            policy_repo_root=args["repo_path"],
            axiom_rules_path=tmp_path / "engine",
            issues=args["issues"],
            local_corpus_release=None,
        )
    assert len(calls) == 1
    assert bool(result) is not failure
    if failure:
        assert args["test_file"].read_bytes() == before
