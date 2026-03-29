"""Tests for model comparison eval helpers."""

import json
from pathlib import Path
from unittest.mock import patch

from autorac.harness.evals import (
    _build_eval_prompt,
    _command_looks_out_of_bounds,
    _hydrate_eval_root,
    evaluate_artifact,
    parse_runner_spec,
    prepare_eval_workspace,
    run_source_eval,
    select_context_files,
)
from autorac.harness.validator_pipeline import ValidationResult


class TestParseRunnerSpec:
    def test_parses_named_runner(self):
        runner = parse_runner_spec("gpt=codex:gpt-5.4")
        assert runner.name == "gpt"
        assert runner.backend == "codex"
        assert runner.model == "gpt-5.4"

    def test_parses_default_name(self):
        runner = parse_runner_spec("claude:opus")
        assert runner.name == "claude-opus"
        assert runner.backend == "claude"
        assert runner.model == "opus"


class TestEvaluateArtifact:
    def test_uses_fallback_source_text_for_grounding(self, tmp_path):
        rac_file = tmp_path / "24" / "a.rac"
        rac_file.parent.mkdir(parents=True)
        rac_file.write_text(
            '"""\n(a) Allowance of credit There shall be allowed a credit of $1,000.\n"""\n'
            "status: encoded\n\n"
            "ctc_amount:\n"
            "    description: \"Child tax credit amount\"\n"
            "    unit: USD\n"
            "    from 2018-01-01: 1000\n"
            "    from 2025-01-01: 2200\n"
        )

        compile_result = ValidationResult("compile", passed=True)
        ci_result = ValidationResult("ci", passed=True)

        with patch(
            "autorac.harness.validator_pipeline.ValidatorPipeline._run_compile_check",
            return_value=compile_result,
        ), patch(
            "autorac.harness.validator_pipeline.ValidatorPipeline._run_ci",
            return_value=ci_result,
        ):
            metrics = evaluate_artifact(
                rac_file=rac_file,
                rac_root=tmp_path,
                rac_path=Path("/tmp/rac"),
                source_text="(a) Allowance of credit There shall be allowed a credit of $1,000.",
            )

        assert metrics.compile_pass
        assert metrics.ci_pass
        assert metrics.grounded_numeric_count == 1
        assert metrics.ungrounded_numeric_count == 1
        assert [item.raw for item in metrics.grounding if not item.grounded] == ["2200"]


class TestRepoAugmentedContext:
    def test_prepare_eval_workspace_allows_arbitrary_identifier_with_explicit_context(
        self, tmp_path
    ):
        repo_root = tmp_path / "repos"
        rac_root = repo_root / "rac"
        rac_root.mkdir(parents=True)
        context_file = repo_root / "rac-us" / "statute" / "26" / "32" / "b" / "2" / "A.rac"
        context_file.parent.mkdir(parents=True)
        context_file.write_text("status: encoded\n")

        runner = parse_runner_spec("codex:gpt-5.4")
        workspace = prepare_eval_workspace(
            citation="9 CCR 2503-6 3.606.1(F)",
            runner=runner,
            output_root=tmp_path / "out",
            source_text="F. Determining Eligibility ... 165 345 518",
            rac_path=rac_root,
            mode="repo-augmented",
            extra_context_paths=[context_file],
        )

        manifest = json.loads(workspace.manifest_file.read_text())
        assert manifest["mode"] == "repo-augmented"
        assert manifest["context_files"][0]["source_path"] == str(context_file)
        copied = workspace.root / manifest["context_files"][0]["workspace_path"]
        assert copied.exists()

    def test_select_context_files_excludes_target(self, tmp_path):
        rac_us_root = tmp_path / "rac-us" / "statute"
        section_dir = rac_us_root / "26" / "24"
        section_dir.mkdir(parents=True)
        (section_dir / "a.rac").write_text("target")
        (section_dir / "b.rac").write_text("sibling b")
        (section_dir / "c.rac").write_text("sibling c")

        selected = select_context_files("26 USC 24(a)", rac_us_root)

        assert section_dir / "a.rac" not in selected
        assert section_dir / "b.rac" in selected
        assert section_dir / "c.rac" in selected

    def test_prepare_eval_workspace_writes_manifest_and_context(self, tmp_path):
        repo_root = tmp_path / "repos"
        rac_root = repo_root / "rac"
        rac_root.mkdir(parents=True)
        rac_us_root = repo_root / "rac-us" / "statute" / "26" / "24"
        rac_us_root.mkdir(parents=True)
        context_file = rac_us_root / "b.rac"
        context_file.write_text("status: encoded\n")

        runner = parse_runner_spec("codex:gpt-5.4")
        with patch(
            "autorac.harness.evals.select_context_files",
            return_value=[context_file],
        ):
            workspace = prepare_eval_workspace(
                citation="26 USC 24(a)",
                runner=runner,
                output_root=tmp_path / "out",
                source_text="(a) Allowance of credit ... $1,000.",
                rac_path=rac_root,
                mode="repo-augmented",
                extra_context_paths=[],
            )

        manifest = json.loads(workspace.manifest_file.read_text())
        assert manifest["mode"] == "repo-augmented"
        assert manifest["source_file"] == "source.txt"
        assert manifest["context_files"][0]["source_path"] == str(context_file)
        copied = workspace.root / manifest["context_files"][0]["workspace_path"]
        assert copied.exists()

    def test_hydrate_eval_root_copies_context_into_import_tree(self, tmp_path):
        repo_root = tmp_path / "repos"
        rac_root = repo_root / "rac"
        rac_root.mkdir(parents=True)
        rac_us_root = repo_root / "rac-us" / "statute" / "26" / "24"
        rac_us_root.mkdir(parents=True)
        context_file = rac_us_root / "c.rac"
        context_file.write_text("status: stub\n")

        runner = parse_runner_spec("codex:gpt-5.4")
        with patch(
            "autorac.harness.evals.select_context_files",
            return_value=[context_file],
        ):
            workspace = prepare_eval_workspace(
                citation="26 USC 24(a)",
                runner=runner,
                output_root=tmp_path / "out",
                source_text="(a) Allowance of credit ... $1,000.",
                rac_path=rac_root,
                mode="repo-augmented",
                extra_context_paths=[],
            )

        eval_root = tmp_path / "eval-root"
        _hydrate_eval_root(eval_root, workspace)

        assert (eval_root / "26" / "24" / "c.rac").read_text() == "status: stub\n"

    def test_prepare_eval_workspace_expands_transitive_context_imports(self, tmp_path):
        repo_root = tmp_path / "repos"
        rac_root = repo_root / "rac"
        rac_root.mkdir(parents=True)

        section_root = repo_root / "rac-us" / "statute" / "26" / "24"
        section_root.mkdir(parents=True)
        aggregator = section_root / "24.rac"
        aggregator.write_text(
            "section_24_credit:\n"
            "    imports:\n"
            "        - 26/24/a#ctc_allowance\n"
            "        - 26/24/c#qualifying_child_count\n"
            "    entity: TaxUnit\n"
            "    period: Year\n"
            "    dtype: Money\n"
        )
        selected = section_root / "c.rac"
        selected.write_text(
            "qualifying_child_count:\n"
            "    imports:\n"
            "        - 26/24/c/2#ctc_meets_citizenship_requirement\n"
            "        - 26/152/c#qualifying_child_of_taxpayer\n"
            "    entity: TaxUnit\n"
            "    period: Year\n"
            "    dtype: Integer\n"
        )

        dep_local = section_root / "c" / "2.rac"
        dep_local.parent.mkdir(parents=True)
        dep_local.write_text("status: encoded\n")

        dep_cross_section = repo_root / "rac-us" / "statute" / "26" / "152" / "c.rac"
        dep_cross_section.parent.mkdir(parents=True)
        dep_cross_section.write_text("status: encoded\n")

        runner = parse_runner_spec("codex:gpt-5.4")
        with patch(
            "autorac.harness.evals.select_context_files",
            return_value=[aggregator, selected],
        ):
            workspace = prepare_eval_workspace(
                citation="26 USC 24(a)",
                runner=runner,
                output_root=tmp_path / "out",
                source_text="(a) Allowance of credit ... $1,000.",
                rac_path=rac_root,
                mode="repo-augmented",
                extra_context_paths=[],
            )

        manifest = json.loads(workspace.manifest_file.read_text())
        copied_sources = {
            item["source_path"]: item["kind"] for item in manifest["context_files"]
        }

        assert copied_sources[str(selected)] == "implementation_precedent"
        assert copied_sources[str(dep_local)] == "implementation_dependency"
        assert copied_sources[str(dep_cross_section)] == "implementation_dependency"
        assert str(section_root / "a.rac") not in copied_sources

        eval_root = tmp_path / "eval-root"
        _hydrate_eval_root(eval_root, workspace)
        assert (eval_root / "26" / "24" / "c" / "2.rac").read_text() == "status: encoded\n"
        assert (eval_root / "26" / "152" / "c.rac").read_text() == "status: encoded\n"

    def test_prompt_includes_scaffold_dates_from_context(self, tmp_path):
        repo_root = tmp_path / "repos"
        rac_root = repo_root / "rac"
        rac_root.mkdir(parents=True)
        rac_us_root = repo_root / "rac-us" / "statute" / "26" / "24"
        rac_us_root.mkdir(parents=True)
        context_file = rac_us_root / "b.rac"
        context_file.write_text(
            "status: encoded\n\n"
            "threshold:\n"
            "    from 1998-01-01: 1000\n"
            "    from 2018-01-01: 2000\n"
        )

        runner = parse_runner_spec("codex:gpt-5.4")
        with patch(
            "autorac.harness.evals.select_context_files",
            return_value=[context_file],
        ):
            workspace = prepare_eval_workspace(
                citation="26 USC 24(a)",
                runner=runner,
                output_root=tmp_path / "out",
                source_text="(a) Allowance of credit ... $1,000.",
                rac_path=rac_root,
                mode="repo-augmented",
                extra_context_paths=[],
            )

        prompt = _build_eval_prompt(
            "26 USC 24(a)",
            "repo-augmented",
            workspace,
            workspace.context_files,
        )

        assert "`1998-01-01`" in prompt
        assert "`2018-01-01`" in prompt
        assert "Prefer the earliest scaffold date" in prompt


class TestUnexpectedAccessDetection:
    def test_flags_parent_directory_traversal(self, tmp_path):
        assert _command_looks_out_of_bounds("bash -lc 'find .. -name *.rac'", tmp_path)

    def test_allows_workspace_paths(self, tmp_path):
        local = tmp_path / "context" / "b.rac"
        local.parent.mkdir(parents=True)
        local.write_text("status: encoded\n")


class TestSourceEval:
    def test_run_source_eval_uses_explicit_context_without_statute_lookup(self, tmp_path):
        rac_root = tmp_path / "rac"
        rac_root.mkdir()
        context_file = tmp_path / "examples" / "piecewise.rac"
        context_file.parent.mkdir(parents=True)
        context_file.write_text("status: encoded\n")

        with patch(
            "autorac.harness.evals._run_prompt_eval",
        ) as mock_prompt_eval, patch(
            "autorac.harness.evals.evaluate_artifact",
        ) as mock_evaluate_artifact:
            mock_prompt_eval.return_value.text = (
                "=== FILE: 9-CCR-2503-6-3.606.1-F.rac ===\n"
                '"""\nF. Determining Eligibility ...\n"""\n'
                "status: encoded\n"
                "grant_standard:\n"
                "    entity: TaxUnit\n"
                "    period: Month\n"
                "    dtype: Money\n"
                "    from 2024-07-01: 165\n"
                "=== FILE: 9-CCR-2503-6-3.606.1-F.rac.test ===\n"
                "grant_standard:\n"
                '  - name: "base case"\n'
                "    period: 2024-07\n"
                "    inputs: {}\n"
                "    expect: 165\n"
            )
            mock_prompt_eval.return_value.duration_ms = 123
            mock_prompt_eval.return_value.tokens = None
            mock_prompt_eval.return_value.estimated_cost_usd = None
            mock_prompt_eval.return_value.actual_cost_usd = None
            mock_prompt_eval.return_value.trace = {}
            mock_prompt_eval.return_value.unexpected_accesses = []
            mock_prompt_eval.return_value.error = None

            mock_evaluate_artifact.return_value = None

            results = run_source_eval(
                source_id="9 CCR 2503-6 3.606.1(F)",
                source_text="F. Determining Eligibility ... 165",
                runner_specs=["codex:gpt-5.4"],
                output_root=tmp_path / "out",
                rac_path=rac_root,
                mode="repo-augmented",
                extra_context_paths=[context_file],
            )

        assert len(results) == 1
        result = results[0]
        assert result.success is True
        assert Path(result.output_file).exists()
        assert Path(result.output_file).with_suffix(".rac.test").exists()
        assert result.retrieved_files == [str(context_file)]

        prompt = mock_prompt_eval.call_args.args[2]
        assert ".rac.test" in prompt
        assert "=== FILE:" in prompt

    def test_allows_relative_workspace_reads(self, tmp_path):
        (tmp_path / "source.txt").write_text("text\n")
        command = "bash -lc 'cat ./source.txt && sed -n \"1,40p\" context/26/24/b.rac'"
        assert not _command_looks_out_of_bounds(command, tmp_path)
