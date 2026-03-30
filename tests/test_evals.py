"""Tests for model comparison eval helpers."""

import json
from pathlib import Path
from unittest.mock import patch

from autorac.harness.evals import (
    _build_eval_prompt,
    _clean_generated_file_content,
    _command_looks_out_of_bounds,
    _fetch_legislation_gov_uk_document,
    _hydrate_eval_root,
    _materialize_eval_artifact,
    _normalize_legislation_gov_uk_source_ref,
    _resolve_akn_section_eid,
    evaluate_artifact,
    extract_akn_section_text,
    parse_runner_spec,
    prepare_eval_workspace,
    run_legislation_gov_uk_section_eval,
    run_akn_section_eval,
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


class TestGeneratedBundleCleaning:
    def test_clean_generated_file_content_strips_fence_and_trailing_prose(self):
        content = (
            "```yaml\n"
            "- name: base\n"
            "  output:\n"
            "    child_benefit_enhanced_rate: 26.05\n"
            "```\n\n"
            "The encoding captures the enhanced rate."
        )

        cleaned = _clean_generated_file_content(content)

        assert cleaned == (
            "- name: base\n"
            "  output:\n"
            "    child_benefit_enhanced_rate: 26.05\n"
        )

    def test_materialize_eval_artifact_cleans_bundled_fences(self, tmp_path):
        output_file = tmp_path / "source" / "uksi-2006-965-regulation-2.rac"
        llm_response = (
            "=== FILE: uksi-2006-965-regulation-2.rac ===\n"
            "```\n"
            "child_benefit_enhanced_rate:\n"
            "    entity: Person\n"
            "    period: Week\n"
            "    dtype: Money\n"
            "```\n"
            "=== FILE: uksi-2006-965-regulation-2.rac.test ===\n"
            "```yaml\n"
            "- name: base\n"
            "  output:\n"
            "    child_benefit_enhanced_rate: 26.05\n"
            "```\n\n"
            "Trailing prose.\n"
        )

        wrote = _materialize_eval_artifact(llm_response, output_file)

        assert wrote is True
        assert output_file.read_text() == (
            "child_benefit_enhanced_rate:\n"
            "    entity: Person\n"
            "    period: Week\n"
            "    dtype: Money\n"
        )
        assert output_file.with_suffix(".rac.test").read_text() == (
            "- name: base\n"
            "  output:\n"
            "    child_benefit_enhanced_rate: 26.05\n"
        )

    def test_can_include_policyengine_metrics_for_uk_artifact(self, tmp_path):
        rac_file = tmp_path / "source" / "uksi-2006-965-regulation-2.rac"
        rac_file.parent.mkdir(parents=True)
        rac_file.write_text(
            '"""\nhttps://www.legislation.gov.uk/uksi/2006/965/regulation/2\n"""\n'
            "status: encoded\n"
        )

        compile_result = ValidationResult("compile", passed=True)
        ci_result = ValidationResult("ci", passed=True)
        pe_result = ValidationResult(
            "policyengine",
            passed=True,
            score=1.0,
            issues=[],
        )

        with patch(
            "autorac.harness.validator_pipeline.ValidatorPipeline._run_compile_check",
            return_value=compile_result,
        ), patch(
            "autorac.harness.validator_pipeline.ValidatorPipeline._run_ci",
            return_value=ci_result,
        ), patch(
            "autorac.harness.validator_pipeline.ValidatorPipeline._run_policyengine",
            return_value=pe_result,
        ) as mock_policyengine:
            metrics = evaluate_artifact(
                rac_file=rac_file,
                rac_root=tmp_path,
                rac_path=Path("/tmp/rac"),
                source_text="26.05",
                oracle="policyengine",
                policyengine_country="uk",
            )

        assert metrics.compile_pass
        assert metrics.ci_pass
        assert metrics.policyengine_pass is True
        assert metrics.policyengine_score == 1.0
        assert metrics.policyengine_issues == []
        mock_policyengine.assert_called_once()


class TestAknSectionEval:
    def test_extract_akn_section_text_supports_standard_akn_subsection_tags(self, tmp_path):
        akn_file = tmp_path / "doc.xml"
        akn_file.write_text(
            """
<akomaNtoso xmlns="http://docs.oasis-open.org/legaldocml/ns/akn/3.0">
  <act>
    <body>
      <section eId="section-1">
        <num>1</num>
        <heading>Example section</heading>
        <subsection eId="section-1-1">
          <num>(1)</num>
          <content>
            <p>Example subsection text.</p>
          </content>
        </subsection>
      </section>
    </body>
  </act>
</akomaNtoso>
            """.strip()
        )

        text = extract_akn_section_text(akn_file, "section-1-1")

        assert "1 Example section" in text
        assert "(1)" in text
        assert "Example subsection text." in text

    def test_extract_akn_section_text_includes_parent_intro_for_leaf_levels(self, tmp_path):
        akn_file = tmp_path / "doc.xml"
        akn_file.write_text(
            """
<akomaNtoso xmlns="http://docs.oasis-open.org/legaldocml/ns/akn/3.0">
  <act>
    <body>
      <section eId="section-1">
        <num>1</num>
        <heading>Example section</heading>
        <subsection eId="section-1-1">
          <num>(1)</num>
          <intro>
            <p>Lead-in text for child levels.</p>
          </intro>
          <level eId="section-1-1-a">
            <num>(a)</num>
            <content>
              <p>Leaf paragraph text.</p>
            </content>
          </level>
        </subsection>
      </section>
    </body>
  </act>
</akomaNtoso>
            """.strip()
        )

        text = extract_akn_section_text(akn_file, "section-1-1-a")

        assert "1 Example section" in text
        assert "(1)" in text
        assert "Lead-in text for child levels." in text
        assert "(a)" in text
        assert "Leaf paragraph text." in text

    def test_extract_akn_section_text_includes_heading_and_table_rows(self, tmp_path):
        akn_file = tmp_path / "doc.xml"
        akn_file.write_text(
            """
<akomaNtoso xmlns="http://docs.oasis-open.org/legaldocml/ns/akn/3.0">
  <doc>
    <mainBody>
      <hcontainer name="section" eId="sec_3_606_1">
        <num>3.606.1</num>
        <heading>Basic Cash Assistance</heading>
        <content>
          <p>A. Payment of Basic Cash Assistance Grants</p>
          <table>
            <tr><th>Children</th><th>Grant</th></tr>
            <tr><td>0</td><td>165</td></tr>
            <tr><td>1</td><td>345</td></tr>
          </table>
          <p>G. Pregnancy allowance text.</p>
        </content>
      </hcontainer>
    </mainBody>
  </doc>
</akomaNtoso>
            """.strip()
        )

        text = extract_akn_section_text(akn_file, "sec_3_606_1")

        assert "3.606.1 Basic Cash Assistance" in text
        assert "A. Payment of Basic Cash Assistance Grants" in text
        assert "Structured table:" in text
        assert "Children | Grant" in text
        assert "0 | 165" in text
        assert "G. Pregnancy allowance text." in text

    def test_extract_akn_section_text_includes_editorial_effective_date(self, tmp_path):
        akn_file = tmp_path / "doc.xml"
        akn_file.write_text(
            """
<akomaNtoso xmlns="http://docs.oasis-open.org/legaldocml/ns/akn/3.0">
  <doc>
    <mainBody>
      <hcontainer name="section" eId="sec_3_606_1_f">
        <num>F</num>
        <heading>Grant Standard</heading>
        <remark status="editorial">Effective date: 2024-07-01</remark>
        <content>
          <p>Grant table text.</p>
        </content>
      </hcontainer>
    </mainBody>
  </doc>
</akomaNtoso>
            """.strip()
        )

        text = extract_akn_section_text(akn_file, "sec_3_606_1_f")

        assert "F Grant Standard" in text
        assert "Effective date: 2024-07-01" in text
        assert "Grant table text." in text

    def test_extract_akn_section_text_includes_expression_valid_from_date(self, tmp_path):
        akn_file = tmp_path / "doc.xml"
        akn_file.write_text(
            """
<akomaNtoso xmlns="http://docs.oasis-open.org/legaldocml/ns/akn/3.0">
  <act>
    <meta>
      <identification source="#">
        <FRBRExpression>
          <FRBRdate date="2025-04-07" name="validFrom"/>
        </FRBRExpression>
      </identification>
    </meta>
    <body>
      <section eId="section-1">
        <num>1</num>
        <heading>Example</heading>
        <content>
          <p>Current amount is 26.05.</p>
        </content>
      </section>
    </body>
  </act>
</akomaNtoso>
            """.strip()
        )

        text = extract_akn_section_text(akn_file, "section-1")

        assert "Editorial note: current text valid from 2025-04-07." in text
        assert "Current amount is 26.05." in text

    def test_run_akn_section_eval_uses_extracted_section_text(self, tmp_path):
        akn_file = tmp_path / "doc.xml"
        akn_file.write_text(
            """
<akomaNtoso xmlns="http://docs.oasis-open.org/legaldocml/ns/akn/3.0">
  <doc>
    <mainBody>
      <hcontainer name="section" eId="sec_3_606_1">
        <num>3.606.1</num>
        <heading>Basic Cash Assistance</heading>
        <content>
          <p>A. Payment of Basic Cash Assistance Grants</p>
        </content>
      </hcontainer>
    </mainBody>
  </doc>
</akomaNtoso>
            """.strip()
        )

        with patch(
            "autorac.harness.evals.run_source_eval",
            return_value=["ok"],
        ) as mock_run_source_eval:
            results = run_akn_section_eval(
                source_id="9 CCR 2503-6 3.606.1",
                akn_file=akn_file,
                section_eid="sec_3_606_1",
                runner_specs=["codex:gpt-5.4"],
                output_root=tmp_path / "out",
                rac_path=tmp_path / "rac",
                mode="cold",
                extra_context_paths=[],
            )

        assert results == ["ok"]
        mock_run_source_eval.assert_called_once()
        assert (
            mock_run_source_eval.call_args.kwargs["source_text"]
            == "3.606.1 Basic Cash Assistance\n\nA. Payment of Basic Cash Assistance Grants"
        )
        assert mock_run_source_eval.call_args.kwargs["oracle"] == "none"
        assert mock_run_source_eval.call_args.kwargs["policyengine_country"] == "auto"

    def test_run_akn_section_eval_rejects_parent_section_by_default(self, tmp_path):
        akn_file = tmp_path / "doc.xml"
        akn_file.write_text(
            """
<akomaNtoso xmlns="http://docs.oasis-open.org/legaldocml/ns/akn/3.0">
  <doc>
    <mainBody>
      <hcontainer name="section" eId="sec_3_606_1">
        <num>3.606.1</num>
        <heading>Basic Cash Assistance</heading>
        <hcontainer name="section" eId="sec_3_606_1_f">
          <num>F</num>
          <content><p>Grant standard table text.</p></content>
        </hcontainer>
        <hcontainer name="section" eId="sec_3_606_1_g">
          <num>G</num>
          <content><p>Pregnancy allowance text.</p></content>
        </hcontainer>
      </hcontainer>
    </mainBody>
  </doc>
</akomaNtoso>
            """.strip()
        )

        with patch("autorac.harness.evals.run_source_eval") as mock_run_source_eval:
            try:
                run_akn_section_eval(
                    source_id="9 CCR 2503-6 3.606.1",
                    akn_file=akn_file,
                    section_eid="sec_3_606_1",
                    runner_specs=["codex:gpt-5.4"],
                    output_root=tmp_path / "out",
                    rac_path=tmp_path / "rac",
                    mode="cold",
                    extra_context_paths=[],
                )
            except ValueError as exc:
                message = str(exc)
            else:
                raise AssertionError("Expected parent-section guardrail to raise")

        assert "Choose an atomic child section instead" in message
        assert "sec_3_606_1_f" in message
        assert "sec_3_606_1_g" in message
        mock_run_source_eval.assert_not_called()

    def test_resolve_akn_section_eid_allows_parent_when_opted_in(self, tmp_path):
        akn_file = tmp_path / "doc.xml"
        akn_file.write_text(
            """
<akomaNtoso xmlns="http://docs.oasis-open.org/legaldocml/ns/akn/3.0">
  <doc>
    <mainBody>
      <hcontainer name="section" eId="sec_3_606_1">
        <num>3.606.1</num>
        <heading>Basic Cash Assistance</heading>
        <hcontainer name="section" eId="sec_3_606_1_f">
          <num>F</num>
          <content><p>Grant standard table text.</p></content>
        </hcontainer>
      </hcontainer>
    </mainBody>
  </doc>
</akomaNtoso>
            """.strip()
        )

        assert (
            _resolve_akn_section_eid(
                akn_file,
                "sec_3_606_1",
                allow_parent=True,
            )
            == "sec_3_606_1"
        )

    def test_resolve_akn_section_eid_rejects_standard_akn_parent_sections(self, tmp_path):
        akn_file = tmp_path / "doc.xml"
        akn_file.write_text(
            """
<akomaNtoso xmlns="http://docs.oasis-open.org/legaldocml/ns/akn/3.0">
  <act>
    <body>
      <section eId="section-1">
        <num>1</num>
        <heading>Example section</heading>
        <subsection eId="section-1-1">
          <num>(1)</num>
          <content><p>Example subsection text.</p></content>
        </subsection>
      </section>
    </body>
  </act>
</akomaNtoso>
            """.strip()
        )

        try:
            _resolve_akn_section_eid(akn_file, "section-1")
        except ValueError as exc:
            message = str(exc)
        else:
            raise AssertionError("Expected parent-section guardrail to raise")

        assert "Choose an atomic child section instead" in message
        assert "section-1-1" in message


class TestUkLegislationFetch:
    def test_normalize_legislation_gov_uk_source_ref_strips_data_extension(self):
        source_id, content_url = _normalize_legislation_gov_uk_source_ref(
            "https://www.legislation.gov.uk/ukpga/2010/1/section/1/data.akn"
        )

        assert source_id == "ukpga/2010/1/section/1"
        assert content_url == "https://www.legislation.gov.uk/ukpga/2010/1/section/1"

    def test_fetch_legislation_gov_uk_document_writes_akn_and_clml(self, tmp_path):
        class FakeResponse:
            def __init__(self, text: str):
                self.text = text

            def raise_for_status(self):
                return None

        with patch(
            "requests.get",
            side_effect=[
                FakeResponse("<akomaNtoso/>"),
                FakeResponse("<Legislation/>"),
            ],
        ) as mock_get:
            fetched = _fetch_legislation_gov_uk_document(
                "https://www.legislation.gov.uk/ukpga/2010/1/section/1",
                tmp_path,
            )

        assert fetched.source_id == "ukpga/2010/1/section/1"
        assert fetched.akn_file.read_text() == "<akomaNtoso/>"
        assert fetched.clml_file.read_text() == "<Legislation/>"
        assert mock_get.call_args_list[0].args[0].endswith("/data.akn")
        assert mock_get.call_args_list[1].args[0].endswith("/data.xml")

    def test_run_legislation_gov_uk_section_eval_uses_primary_akn_node_when_unspecified(
        self, tmp_path
    ):
        fetched_dir = tmp_path / "fetched"
        fetched_dir.mkdir()
        akn_file = fetched_dir / "source.akn"
        clml_file = fetched_dir / "source.xml"
        akn_file.write_text(
            """
<akomaNtoso xmlns="http://docs.oasis-open.org/legaldocml/ns/akn/3.0">
  <act>
    <body>
      <section eId="section-1">
        <num>1</num>
        <heading>Example section</heading>
        <subsection eId="section-1-1">
          <num>(1)</num>
          <content><p>Example subsection text.</p></content>
        </subsection>
      </section>
    </body>
  </act>
</akomaNtoso>
            """.strip()
        )
        clml_file.write_text("<Legislation/>")

        with patch(
            "autorac.harness.evals._fetch_legislation_gov_uk_document"
        ) as mock_fetch, patch(
            "autorac.harness.evals.run_akn_section_eval",
            return_value=["ok"],
        ) as mock_run:
            mock_fetch.return_value.source_id = "ukpga/2010/1/section/1"
            mock_fetch.return_value.content_url = (
                "https://www.legislation.gov.uk/ukpga/2010/1/section/1"
            )
            mock_fetch.return_value.akn_file = akn_file
            mock_fetch.return_value.clml_file = clml_file

            results = run_legislation_gov_uk_section_eval(
                source_ref="https://www.legislation.gov.uk/ukpga/2010/1/section/1",
                runner_specs=["codex:gpt-5.4"],
                output_root=tmp_path / "out",
                rac_path=tmp_path / "rac",
                mode="cold",
                extra_context_paths=[],
            )

        assert results == ["ok"]
        mock_run.assert_called_once()
        assert mock_run.call_args.kwargs["section_eid"] == "section-1"
        assert mock_run.call_args.kwargs["source_id"] == "ukpga/2010/1/section/1"
        assert mock_run.call_args.kwargs["oracle"] == "policyengine"
        assert mock_run.call_args.kwargs["policyengine_country"] == "uk"


class TestEvalPrompt:
    def test_build_eval_prompt_includes_rac_syntax_guardrails(self, tmp_path):
        workspace = prepare_eval_workspace(
            citation="9 CCR 2503-6 3.606.1",
            runner=parse_runner_spec("codex:gpt-5.4"),
            output_root=tmp_path / "out",
            source_text="Grant standard is 165 for one child.",
            rac_path=tmp_path / "rac",
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "9 CCR 2503-6 3.606.1",
            "cold",
            workspace,
            [],
            target_file_name="9-CCR-2503-6-3.606.1.rac",
            include_tests=True,
        )

        assert "Do not invent schema keys like `namespace:`, `parameter`, `variable`, or `rule:`." in prompt
        assert "entity:" in prompt
        assert "period:" in prompt
        assert "dtype:" in prompt

    def test_build_eval_prompt_includes_supported_schema_enums(self, tmp_path):
        workspace = prepare_eval_workspace(
            citation="uksi/2006/965/regulation/2",
            runner=parse_runner_spec("codex:gpt-5.4"),
            output_root=tmp_path / "out",
            source_text="2. Rate of child benefit ... 25.60 ... 16.95",
            rac_path=tmp_path / "rac",
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "uksi/2006/965/regulation/2",
            "cold",
            workspace,
            [],
            target_file_name="uksi-2006-965-regulation-2.rac",
            include_tests=True,
        )

        assert "Do not invent new entities, periods, or dtypes." in prompt
        assert "Allowed `entity:` values are `Person`, `TaxUnit`, `Household`, `Family`" in prompt
        assert "Allowed `period:` values are `Year`, `Month`, `Week`, `Day`." in prompt
        assert (
            "Allowed `dtype:` values are `Money`, `Rate`, `Boolean`, `Integer`, "
            "`Count`, `String`, `Decimal`, `Float`, or `Enum[Name]`."
        ) in prompt

    def test_build_eval_prompt_includes_unsupported_entity_fallback(self, tmp_path):
        workspace = prepare_eval_workspace(
            citation="ukpga/2010/1/section/1",
            runner=parse_runner_spec("codex:gpt-5.4"),
            output_root=tmp_path / "out",
            source_text="(a) cease to be in force",
            rac_path=tmp_path / "rac",
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "ukpga/2010/1/section/1",
            "cold",
            workspace,
            [],
            target_file_name="ukpga-2010-1-section-1.rac",
            include_tests=True,
        )

        assert "If the source cannot be represented faithfully with the supported schema" in prompt
        assert "`status: entity_not_supported`" in prompt
        assert "`status: deferred`" in prompt

    def test_build_eval_prompt_forbids_python_inline_ternaries(self, tmp_path):
        workspace = prepare_eval_workspace(
            citation="uksi/2006/965/regulation/2",
            runner=parse_runner_spec("codex:gpt-5.4"),
            output_root=tmp_path / "out",
            source_text="2. Rate of child benefit ... 26.05",
            rac_path=tmp_path / "rac",
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "uksi/2006/965/regulation/2",
            "cold",
            workspace,
            [],
            target_file_name="uksi-2006-965-regulation-2.rac",
            include_tests=True,
        )

        assert "Do not use Python inline ternaries" in prompt
        assert "`x if cond else y`" in prompt

    def test_build_eval_prompt_requires_rac_conditional_expression_syntax(
        self, tmp_path
    ):
        workspace = prepare_eval_workspace(
            citation="uksi/2006/965/regulation/2",
            runner=parse_runner_spec("codex:gpt-5.4"),
            output_root=tmp_path / "out",
            source_text="2. Rate of child benefit ... 26.05",
            rac_path=tmp_path / "rac",
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "uksi/2006/965/regulation/2",
            "cold",
            workspace,
            [],
            target_file_name="uksi-2006-965-regulation-2.rac",
            include_tests=True,
        )

        assert "`if condition: value else: other_value`" in prompt
        assert "Do not use YAML-style `if:` / `then:` / `else:` blocks." in prompt

    def test_build_eval_prompt_for_uk_leaf_prefers_person_over_family_constant(
        self, tmp_path
    ):
        workspace = prepare_eval_workspace(
            citation="uksi/2006/965/regulation/2",
            runner=parse_runner_spec("claude:opus"),
            output_root=tmp_path / "out",
            source_text=(
                "Editorial note: current text valid from 2025-04-07.\n\n"
                "The weekly rate of child benefit payable in respect of a child "
                "or qualifying young person shall be 26.05."
            ),
            rac_path=tmp_path / "rac",
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "uksi/2006/965/regulation/2",
            "cold",
            workspace,
            [],
            target_file_name="uksi-2006-965-regulation-2.rac",
            include_tests=True,
        )

        assert 'Prefer `Person` when the source states an amount or condition "in respect of"' in prompt
        assert "do not collapse it into an unconditional family-level constant" in prompt

    def test_build_eval_prompt_for_uk_leaf_forbids_speculative_future_period_tests(
        self, tmp_path
    ):
        workspace = prepare_eval_workspace(
            citation="uksi/2006/965/regulation/2",
            runner=parse_runner_spec("claude:opus"),
            output_root=tmp_path / "out",
            source_text="Editorial note: current text valid from 2025-04-07.\n26.05",
            rac_path=tmp_path / "rac",
            mode="cold",
            extra_context_paths=[],
        )

        prompt = _build_eval_prompt(
            "uksi/2006/965/regulation/2",
            "cold",
            workspace,
            [],
            target_file_name="uksi-2006-965-regulation-2.rac",
            include_tests=True,
        )

        assert "The `.rac.test` file must contain YAML only" in prompt
        assert "Do not add speculative future-period tests" in prompt
        assert "must vary a real legal condition" in prompt
        assert "must contain factual predicates or quantities, not the output variable" in prompt


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
            "a.rac",
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

    def test_run_source_eval_passes_oracle_settings_to_evaluate_artifact(self, tmp_path):
        rac_root = tmp_path / "rac"
        rac_root.mkdir()

        with patch(
            "autorac.harness.evals._run_prompt_eval",
        ) as mock_prompt_eval, patch(
            "autorac.harness.evals.evaluate_artifact",
        ) as mock_evaluate_artifact:
            mock_prompt_eval.return_value.text = (
                "=== FILE: uksi-2006-965-regulation-2.rac ===\n"
                '"""\nhttps://www.legislation.gov.uk/uksi/2006/965/regulation/2\n"""\n'
                "status: encoded\n"
                "=== FILE: uksi-2006-965-regulation-2.rac.test ===\n"
                "- name: base\n"
                "  input: {}\n"
                "  output:\n"
                "    child_benefit_enhanced_rate: 26.05\n"
            )
            mock_prompt_eval.return_value.duration_ms = 123
            mock_prompt_eval.return_value.tokens = None
            mock_prompt_eval.return_value.estimated_cost_usd = None
            mock_prompt_eval.return_value.actual_cost_usd = None
            mock_prompt_eval.return_value.trace = {}
            mock_prompt_eval.return_value.unexpected_accesses = []
            mock_prompt_eval.return_value.error = None
            mock_evaluate_artifact.return_value = None

            run_source_eval(
                source_id="uksi/2006/965/regulation/2",
                source_text="26.05",
                runner_specs=["codex:gpt-5.4"],
                output_root=tmp_path / "out",
                rac_path=rac_root,
                mode="cold",
                oracle="policyengine",
                policyengine_country="uk",
            )

        assert mock_evaluate_artifact.call_args.kwargs["oracle"] == "policyengine"
        assert (
            mock_evaluate_artifact.call_args.kwargs["policyengine_country"] == "uk"
        )

    def test_allows_relative_workspace_reads(self, tmp_path):
        (tmp_path / "source.txt").write_text("text\n")
        command = "bash -lc 'cat ./source.txt && sed -n \"1,40p\" context/26/24/b.rac'"
        assert not _command_looks_out_of_bounds(command, tmp_path)
