"""Tests for the EncodeBench verifier track (``benchmarks/verifier``).

No test here touches the network: the referee runner is exercised through a
fake ``JudgeClient``, the Jev runner through a fake TypeSafe client, and the
end-to-end CLI path through the replay runner.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
VERIFIER_ROOT = REPO_ROOT / "benchmarks" / "verifier"
if str(VERIFIER_ROOT) not in sys.path:
    sys.path.insert(0, str(VERIFIER_ROOT))

from encodebench_verifier import (  # noqa: E402
    DEFECT_KINDS,
    RESULTS_SCHEMA,
    SUITE_SCHEMA,
    VARIANT_CONTROL,
    VARIANT_DEFECTIVE,
)
from encodebench_verifier import board as board_module  # noqa: E402
from encodebench_verifier import cli as verifier_cli  # noqa: E402
from encodebench_verifier.canonical import (  # noqa: E402
    canonical_json_sha256,
    dump_yaml_document,
    load_yaml_document,
)
from encodebench_verifier.cases import (  # noqa: E402
    CaseSuite,
    Locator,
    SuiteError,
    VerifierCase,
)
from encodebench_verifier.judges import (  # noqa: E402
    REFEREE_KIND_MAP,
    JevRunner,
    RefereeRunner,
    ReplayRunner,
    make_runner,
)
from encodebench_verifier.judges.base import (  # noqa: E402
    CHANNEL_NATIVE,
    CHANNEL_VERDICT_FALLBACK,
    JudgeResponse,
)
from encodebench_verifier.judges.jev import (  # noqa: E402
    KIND_NOULS,
    build_questions,
    build_state,
)
from encodebench_verifier.judges.referee import verdict_score_from  # noqa: E402
from encodebench_verifier.localization import localize  # noqa: E402
from encodebench_verifier.metrics import (  # noqa: E402
    auc,
    detection_at_false_alarm_ceiling,
    paired_rise_rate,
)
from encodebench_verifier.mutator import (  # noqa: E402
    MUTATOR_VERSION,
    MutationError,
    leaf_differences,
    mutate,
)
from encodebench_verifier.pricing import Price, cost_usd, load_pricing  # noqa: E402
from encodebench_verifier.results import (  # noqa: E402
    ResultsError,
    load_results,
    run_suite,
)
from encodebench_verifier.sources import KnownGoodArtifact  # noqa: E402
from encodebench_verifier.sources import real as real_source  # noqa: E402
from encodebench_verifier.sources.synthetic import build_synthetic_suite  # noqa: E402

from axiom_encode.judges import JudgeCall, TokenCounts, statutory_fidelity  # noqa: E402
from axiom_encode.judges.client import truncate_provision  # noqa: E402

FIXTURE_REAL = VERIFIER_ROOT / "fixtures" / "real_defects_example"

PROVISION = (
    "(d) An individual whose tax table income for the taxable year does not "
    "exceed $60,000 shall pay the tax shown in tables prescribed by the "
    "Director, but in any event not less than $20,000. This subsection shall "
    "not apply to an estate or trust. Applies to taxable years beginning "
    "after December 31, 2025. A household member who is an employee and a "
    "resident qualifies."
)

ARTIFACT = """format: rulespec/v1
module:
  proof_validation:
    required: true
  source_verification:
    corpus_citation_path: us-de/statute/30/1102
  summary: |-
    Tax table method.
    Two lines.
rules:
- name: statutory_tax_table_income_ceiling
  kind: parameter
  dtype: Money
  unit: USD
  period: Year
  source: 1102(d)
  metadata:
    proof:
      atoms:
      - path: versions[0].formula
        kind: amount
        source:
          corpus_citation_path: us-de/statute/30/1102
          excerpt: does not exceed $60,000
  versions:
  - effective_from: '2026-01-01'
    formula: '60000'
- name: tax_table_method_applies
  kind: derived
  entity: Person
  dtype: Judgment
  period: Year
  source: 1102(d)
  versions:
  - effective_from: '2026-01-01'
    formula: 'tax_table_income <= statutory_tax_table_income_ceiling

      and not individual_is_estate_or_trust and household_member_is_resident'
- name: household_size_ok
  kind: derived
  entity: Household
  dtype: Judgment
  period: Year
  source: 1102(d)
  versions:
  - effective_from: 2026-01-01
    formula: household_size > 1 or household_has_dependent
"""


def _window(text: str) -> str:
    return truncate_provision(text, 24_000)


# -- mutator ---------------------------------------------------------------------


@pytest.mark.parametrize("kind", DEFECT_KINDS)
def test_every_kind_is_a_single_leaf_edit_that_round_trips(kind):
    mutation = mutate(ARTIFACT, _window(PROVISION), kind, rng=random.Random(3))
    assert mutation is not None, kind
    diffs = leaf_differences(mutation.control_document, mutation.defective_document)
    assert len(diffs) == 1, (kind, diffs)
    assert mutation.locator.path.replace("rules[", ".rules[").lstrip(".") in (
        diffs[0].replace("rules[", ".rules[").lstrip("."),
        diffs[0],
    ) or diffs[0].endswith(mutation.locator.path.split(".")[-1])
    # Both variants are canonical YAML that reparses to their documents.
    assert load_yaml_document(mutation.control_text) == mutation.control_document
    assert load_yaml_document(mutation.defective_text) == mutation.defective_document
    assert mutation.control_text != mutation.defective_text
    assert mutation.kind == kind
    assert mutation.locator.rule_name is not None


def test_mutation_is_deterministic_under_seed():
    first = mutate(
        ARTIFACT, _window(PROVISION), "amount_changed", rng=random.Random(11)
    )
    second = mutate(
        ARTIFACT, _window(PROVISION), "amount_changed", rng=random.Random(11)
    )
    assert first.defective_text == second.defective_text
    assert first.locator == second.locator


def test_amount_changed_only_touches_numbers_present_in_the_provision():
    provision = "The rate is 25 percent."  # 60000 and 20000 absent
    mutation = mutate(ARTIFACT, provision, "amount_changed", rng=random.Random(1))
    assert mutation is None
    mutation = mutate(
        ARTIFACT, "the ceiling is $60,000", "amount_changed", rng=random.Random(1)
    )
    assert mutation is not None
    assert mutation.locator.before == "60000"
    assert mutation.locator.after not in ("60000",)
    assert mutation.locator.after not in "60,000".replace(",", "")
    assert "versions[0].formula" in mutation.locator.path


def test_boundary_flip_maps_operators_both_ways():
    seen = set()
    for seed in range(20):
        mutation = mutate(
            ARTIFACT, _window(PROVISION), "boundary_flipped", rng=random.Random(seed)
        )
        seen.add((mutation.locator.before, mutation.locator.after))
    assert ("<=", "<") in seen
    assert (">", ">=") in seen
    assert all(a != b for a, b in seen)


def test_conjunct_dropped_skips_if_else_formulas_and_keeps_valid_expression():
    artifact = ARTIFACT.replace(
        "formula: household_size > 1 or household_has_dependent",
        "formula: 'if a and b: 1 else: 0'",
    )
    for seed in range(10):
        mutation = mutate(
            artifact, _window(PROVISION), "conjunct_dropped", rng=random.Random(seed)
        )
        assert mutation is not None
        # Only the plain conjunction rule may be edited.
        assert mutation.locator.rule_name == "tax_table_method_applies"
        formula = mutation.defective_document["rules"][1]["versions"][0]["formula"]
        assert " and " in formula or "and" not in formula
        assert not formula.startswith("and ") and not formula.endswith(" and")
        assert mutation.locator.token in (
            "tax_table_income",
            "individual_is_estate_or_trust",
            "household_member_is_resident",
        )


def test_polarity_swap_never_touches_identifiers_containing_and_or_or():
    artifact = ARTIFACT.replace("household_has_dependent", "brand_or_ander")
    for seed in range(10):
        mutation = mutate(
            artifact, _window(PROVISION), "polarity_swapped", rng=random.Random(seed)
        )
        rule = mutation.defective_document["rules"][mutation.locator.rule_index]
        text = json.dumps(rule, default=str)
        assert "brand_or_ander" in text or mutation.locator.rule_index != 2
        assert mutation.locator.before in ("and", "or")


def test_date_or_period_requires_the_provision_to_state_the_original():
    no_dates = "An individual pays tax monthly on the amount."
    mutation = mutate(ARTIFACT, no_dates, "date_or_period_wrong", rng=random.Random(1))
    # Provision mentions "month" for a Year rule -> period swap blocked (target
    # period word present); year 2026 absent -> date shift blocked.
    assert mutation is None
    with_year = "For taxable years beginning in 2026 an individual pays annually."
    mutation = mutate(ARTIFACT, with_year, "date_or_period_wrong", rng=random.Random(1))
    assert mutation is not None
    assert mutation.locator.after in ("2027-01-01", "Month")


def test_date_shift_preserves_leaf_type_for_unquoted_dates():
    import datetime

    provision = "Effective for years beginning in 2026."
    found = False
    for seed in range(30):
        mutation = mutate(
            ARTIFACT, provision, "date_or_period_wrong", rng=random.Random(seed)
        )
        if (
            mutation
            and mutation.locator.path.endswith("versions[0].effective_from")
            and (mutation.locator.rule_index == 2)
        ):
            leaf = mutation.defective_document["rules"][2]["versions"][0][
                "effective_from"
            ]
            assert isinstance(leaf, datetime.date)
            found = True
            break
    assert found


def test_entity_wrong_needs_provision_mention_and_avoids_mentioned_replacements():
    mutation = mutate(
        ARTIFACT, "Rates apply to each vessel.", "entity_wrong", rng=random.Random(1)
    )
    assert mutation is None
    mutation = mutate(
        ARTIFACT, "Each individual pays.", "entity_wrong", rng=random.Random(1)
    )
    assert mutation is not None
    assert mutation.locator.before == "Person"
    assert mutation.locator.after != "Person"


def test_mutator_never_edits_metadata_hashes_or_citations():
    for kind in DEFECT_KINDS:
        for seed in range(5):
            mutation = mutate(
                ARTIFACT, _window(PROVISION), kind, rng=random.Random(seed)
            )
            if mutation is None:
                continue
            control = mutation.control_document
            defective = mutation.defective_document
            assert control["module"] == defective["module"]
            for rule_c, rule_d in zip(control["rules"], defective["rules"]):
                assert rule_c.get("metadata") == rule_d.get("metadata")
                assert rule_c.get("source") == rule_d.get("source")
            (diff,) = leaf_differences(control, defective)
            if kind in (
                "amount_changed",
                "boundary_flipped",
                "conjunct_dropped",
                "polarity_swapped",
            ):
                assert diff.endswith(".formula") or diff.endswith(".value")
            elif kind == "date_or_period_wrong":
                assert diff.endswith(".effective_from") or diff.endswith(".period")
            else:
                assert diff.endswith(".entity")


def test_mutator_rejects_artifacts_without_rules():
    with pytest.raises(MutationError):
        mutate(
            "format: rulespec/v1\nrules: []\n",
            PROVISION,
            "amount_changed",
            rng=random.Random(0),
        )
    with pytest.raises(MutationError):
        mutate("not: [valid", PROVISION, "amount_changed", rng=random.Random(0))
    with pytest.raises(ValueError):
        mutate(ARTIFACT, PROVISION, "nonsense", rng=random.Random(0))


def test_canonical_dump_round_trips_multiline_and_quoted_scalars():
    document = load_yaml_document(ARTIFACT)
    text = dump_yaml_document(document)
    assert load_yaml_document(text) == document
    assert (
        "'60000'" in text or '"60000"' in text
    )  # numeric-looking strings stay strings
    assert "|-" in text or "|" in text  # multi-line strings render as blocks


def test_mutator_version_is_pinned():
    assert MUTATOR_VERSION == "1.0.0"


# -- synthetic suite -------------------------------------------------------------


def _artifacts(n: int = 12) -> list[KnownGoodArtifact]:
    items = []
    for i in range(n):
        items.append(
            KnownGoodArtifact(
                key=f"run{i:02d}",
                citation=f"us-de/statute/30/{1100 + i}",
                provision_text=PROVISION,
                artifact_text=ARTIFACT,
                origin={"source": "test", "generator_model": "gpt-5.5"},
            )
        )
    return items


def test_build_synthetic_suite_pairs_controls_and_fills_kinds_by_deficit():
    suite, report = build_synthetic_suite(
        _artifacts(12),
        name="unit",
        source_kind="test",
        source_identity={"n": 12},
        provision_chars=24_000,
        truncate=truncate_provision,
        per_kind=2,
        seed=5,
    )
    assert suite.summary()["pair_count"] == 12
    assert set(suite.summary()["defective_by_kind"].values()) == {2}
    assert report["short_of_quota"] == {}
    pairs = {}
    for case in suite.cases:
        pairs.setdefault(case.pair_id, {})[case.variant] = case
    for members in pairs.values():
        assert set(members) == {VARIANT_CONTROL, VARIANT_DEFECTIVE}
        control, defective = members[VARIANT_CONTROL], members[VARIANT_DEFECTIVE]
        assert control.provision_text == defective.provision_text
        assert control.artifact_text != defective.artifact_text
        assert control.locator is None and defective.locator is not None
        assert control.control_clean == "known_good_gate"
    # One artifact -> one pair: no citation is reused across kinds.
    citations = [c.citation for c in suite.cases if c.is_defective]
    assert len(citations) == len(set(citations))
    assert suite.mutator["version"] == MUTATOR_VERSION


def test_synthetic_suite_is_reproducible_and_seed_sensitive():
    kwargs = dict(
        name="unit",
        source_kind="test",
        source_identity={},
        provision_chars=24_000,
        truncate=truncate_provision,
        per_kind=1,
    )
    a, _ = build_synthetic_suite(_artifacts(8), seed=1, **kwargs)
    b, _ = build_synthetic_suite(_artifacts(8), seed=1, **kwargs)
    c, _ = build_synthetic_suite(_artifacts(8), seed=2, **kwargs)
    assert a.sha256 == b.sha256
    assert a.sha256 != c.sha256


def test_suite_round_trip_and_tamper_detection(tmp_path):
    suite, _ = build_synthetic_suite(
        _artifacts(6),
        name="unit",
        source_kind="test",
        source_identity={},
        provision_chars=24_000,
        truncate=truncate_provision,
        per_kind=1,
        seed=3,
    )
    suite_path, manifest_path = suite.write(tmp_path / "suite")
    loaded = CaseSuite.load(suite_path)
    assert loaded.sha256 == suite.sha256
    manifest = json.loads(manifest_path.read_text())
    assert manifest["sha256"] == suite.sha256
    assert "provision_text" not in json.dumps(manifest)
    payload = json.loads(suite_path.read_text())
    payload["cases"][1]["artifact_text"] += "\n# edited"
    suite_path.write_text(json.dumps(payload))
    with pytest.raises(SuiteError, match="sha256"):
        CaseSuite.load(suite_path)


def test_suite_refuses_unpaired_cases():
    case = VerifierCase(
        pair_id="p",
        variant=VARIANT_CONTROL,
        defect_kind="amount_changed",
        citation="c",
        provision_text="p",
        artifact_text="a",
    )
    payload = CaseSuite(
        name="x", source_kind="t", source_identity={}, cases=[case], provision_chars=10
    ).to_dict()
    with pytest.raises(SuiteError, match="missing a control or defective"):
        CaseSuite.from_dict(payload)
    with pytest.raises(SuiteError, match="no locator"):
        VerifierCase(
            pair_id="p",
            variant=VARIANT_DEFECTIVE,
            defect_kind="amount_changed",
            citation="c",
            provision_text="p",
            artifact_text="a",
        )


# -- real source -----------------------------------------------------------------


def test_real_fixture_loads_as_unverified_controls():
    suite = real_source.build_real_suite(
        FIXTURE_REAL, provision_chars=24_000, truncate=truncate_provision
    )
    assert suite.source_kind == "real_defects"
    assert suite.summary()["pair_count"] == 2
    kinds = {c.pair_id: c.defect_kind for c in suite.cases if c.is_defective}
    assert kinds == {"rd-0001": "boundary_flipped", "rd-0002": "conjunct_dropped"}
    for case in suite.cases:
        assert case.control_clean == "unverified"
    defective = next(
        c for c in suite.cases if c.pair_id == "rd-0001" and c.is_defective
    )
    assert defective.locator.rule_name == "tax_table_method_applies"
    assert defective.locator.token == "tax_table_income"
    control = next(
        c for c in suite.cases if c.pair_id == "rd-0001" and not c.is_defective
    )
    assert "<=" in control.artifact_text and "< statutory" in defective.artifact_text
    assert suite.mutator is None
    assert any("not" in note and "clean" in note for note in suite.notes)


def test_real_loader_refuses_hash_mismatch_and_identical_pairs(tmp_path):
    import shutil

    root = tmp_path / "real"
    shutil.copytree(FIXTURE_REAL, root)
    case_dir = root / "cases" / "rd-0001"
    (case_dir / "pre_fix.yaml").write_text(
        (case_dir / "pre_fix.yaml").read_text() + "# x\n"
    )
    with pytest.raises(real_source.RealDefectsError, match="sha256"):
        real_source.build_real_suite(
            root, provision_chars=24_000, truncate=truncate_provision
        )
    payload = json.loads((case_dir / "case.json").read_text())
    payload.pop("hashes")
    (case_dir / "pre_fix.yaml").write_text((case_dir / "post_fix.yaml").read_text())
    (case_dir / "case.json").write_text(json.dumps(payload))
    with pytest.raises(real_source.RealDefectsError, match="identical"):
        real_source.build_real_suite(
            root, provision_chars=24_000, truncate=truncate_provision
        )


def test_real_loader_maps_unknown_kinds_to_other_and_accepts_inline_content(tmp_path):
    root = tmp_path / "real"
    (root / "c1").mkdir(parents=True)
    (root / "c1" / "case.json").write_text(
        json.dumps(
            {
                "case_id": "c1",
                "defect_kind": "wrong-import",
                "locator": "some_rule.versions[0].formula",
                "provision_text": "text",
                "pre_fix_yaml": "rules: [{name: a, versions: [{formula: '1'}]}]\n",
                "post_fix_yaml": "rules: [{name: a, versions: [{formula: '2'}]}]\n",
            }
        )
    )
    suite = real_source.build_real_suite(
        root, provision_chars=100, truncate=truncate_provision
    )
    defective = next(c for c in suite.cases if c.is_defective)
    assert defective.defect_kind == "other:wrong_import"
    assert defective.locator.rule_name == "some_rule"
    assert real_source.normalise_kind("Boundary Direction") == "boundary_flipped"
    assert real_source.normalise_kind("effective_date") == "date_or_period_wrong"


# -- metrics ---------------------------------------------------------------------


def test_auc_and_detection_at_ceiling_known_values():
    assert auc([0.9, 0.8], [0.1, 0.2]) == 1.0
    assert auc([0.1, 0.2], [0.9, 0.8]) == 0.0
    assert auc([0.5, 0.5], [0.5, 0.5]) == 0.5
    assert auc([], [0.1]) is None
    assert paired_rise_rate([(0.9, 0.1), (0.2, 0.3)]) == 0.5
    # Ceiling 0: detection = share of positives above every negative.
    assert detection_at_false_alarm_ceiling(
        [0.9, 0.6, 0.3], [0.5, 0.2, 0.1], 0.0
    ) == pytest.approx(2 / 3)
    # Ceiling 1/3 with three negatives lets one negative through (threshold 0.2).
    assert (
        detection_at_false_alarm_ceiling([0.9, 0.6, 0.3], [0.5, 0.2, 0.1], 1 / 3) == 1.0
    )
    with pytest.raises(ValueError):
        detection_at_false_alarm_ceiling([0.1], [0.1], 1.5)


def test_referee_verdict_score_uses_confidence_as_probability_of_correctness():
    assert verdict_score_from("flag", 0.9) == 0.9
    assert verdict_score_from("pass", 0.9) == pytest.approx(0.1)
    assert verdict_score_from("flag", None) == 1.0
    assert verdict_score_from("pass", None) == 0.0
    assert verdict_score_from("flag", 7) == 1.0  # clamped


def test_localization_matches_rule_name_index_or_token():
    locator = Locator(
        path="rules[1].versions[0].formula",
        rule_index=1,
        rule_name="tax_table_method_applies",
        token="1000",
    )
    assert (
        localize(locator, [{"rule_path": "tax_table_method_applies.versions[0]"}])[0]
        is True
    )
    assert localize(locator, [{"rule_path": "rules[1]"}])[0] is True
    assert (
        localize(
            locator, [{"rule_path": "other", "explanation": "states 1000 not 800"}]
        )[0]
        is True
    )
    assert localize(locator, [{"rule_path": "other", "explanation": "vague"}]) == (
        False,
        None,
    )
    assert localize(locator, []) == (False, None)
    assert localize(None, [{"rule_path": "x"}]) == (None, None)


def test_pricing_table_has_sources_and_costs_are_blank_without_a_price():
    prices = load_pricing()
    for model, price in prices.items():
        assert price.source, model
        assert price.input_usd_per_million >= 0
    assert "claude-haiku-4-5-20251001" in prices
    assert "jev-1.13.0" in prices
    assert "claude-sonnet-4-5" not in prices
    assert cost_usd(None, 100, 100) is None
    assert cost_usd(Price("m", 1.0, 5.0, "s"), 1_000_000, 200_000) == pytest.approx(2.0)


# -- judges ----------------------------------------------------------------------


class _FakeJudgeClient:
    def __init__(self, payload=None, *, error=None, model="claude-haiku-4-5-20251001"):
        self.payload = payload
        self.error = error
        self.model = model
        self.escalation_model = model
        self.generator_model = "gpt-5.5"
        self.provision_chars = 24_000
        self.calls: list[dict] = []

    def call(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            from axiom_encode.judges import JudgeError

            return JudgeCall(
                payload=None,
                model=self.model,
                escalated=False,
                tokens=TokenCounts(1, 0),
                error=JudgeError(type=self.error, message="boom"),
            )
        return JudgeCall(
            payload=self.payload,
            model=self.model,
            escalated=False,
            tokens=TokenCounts(300, 40),
        )


def _case(variant=VARIANT_DEFECTIVE, kind="amount_changed"):
    return VerifierCase(
        pair_id="p1",
        variant=variant,
        defect_kind=kind,
        citation="us-de/statute/30/1102",
        provision_text=PROVISION,
        artifact_text=ARTIFACT,
        locator=Locator(
            path="rules[0].versions[0].formula",
            rule_index=0,
            rule_name="statutory_tax_table_income_ceiling",
            token="75000",
        )
        if variant == VARIANT_DEFECTIVE
        else None,
        origin={"generator_model": "gpt-5.5"},
    )


def test_referee_runner_uses_incumbent_prompt_and_maps_kinds():
    fake = _FakeJudgeClient(
        {
            "verdict": "flag",
            "confidence": 0.8,
            "findings": [
                {
                    "clause_ref": "(d)",
                    "rule_path": "statutory_tax_table_income_ceiling",
                    "kind": "amount_mismatch",
                    "explanation": "75000 is not in the provision",
                }
            ],
        }
    )
    runner = RefereeRunner("claude-haiku-4-5-20251001", client_factory=lambda gen: fake)
    response = runner.judge(_case())
    assert response.verdict == "flag"
    assert response.verdict_score == 0.8
    assert response.kind_scores["amount_changed"] == 1.0
    assert response.kind_scores["boundary_flipped"] == 0.0
    assert response.kind_score_channels["amount_changed"] == CHANNEL_NATIVE
    assert response.kind_score_channels["entity_wrong"] == CHANNEL_VERDICT_FALLBACK
    assert response.kind_scores["entity_wrong"] == 0.8
    assert response.tokens_input == 300 and response.tokens_output == 40
    # The exact incumbent prompt and schema went over the wire.
    call = fake.calls[0]
    assert call["system"] == statutory_fidelity._SYSTEM
    assert call["schema"] == statutory_fidelity._SCHEMA
    assert call["user_prompt"] == statutory_fidelity.build_prompt(
        truncate_provision(PROVISION, 24_000),
        ARTIFACT,
        citation="us-de/statute/30/1102",
    )
    identity = runner.identity()
    assert identity["escalation"] is False
    assert len(identity["judge_prompt_sha256"]) == 64
    assert set(identity["finding_kind_map"]) == set(DEFECT_KINDS)
    assert REFEREE_KIND_MAP["date_or_period_wrong"] == ()


def test_referee_runner_fails_closed_on_client_error():
    runner = RefereeRunner(
        "claude-haiku-4-5-20251001",
        client_factory=lambda gen: _FakeJudgeClient(error="rate_limit"),
    )
    response = runner.judge(_case())
    assert response.verdict == "error"
    assert response.verdict_score is None
    assert response.error["type"] == "rate_limit"
    assert not response.ok


def test_referee_runner_pass_verdict_scores_low():
    fake = _FakeJudgeClient({"verdict": "pass", "confidence": 0.7, "findings": []})
    runner = RefereeRunner("claude-haiku-4-5-20251001", client_factory=lambda gen: fake)
    response = runner.judge(_case(VARIANT_CONTROL))
    assert response.verdict == "pass"
    assert response.verdict_score == pytest.approx(0.3)
    assert all(v in (0.0, pytest.approx(0.3)) for v in response.kind_scores.values())


class _FakeAnswer:
    def __init__(self, **fields):
        self.__dict__.update(fields)


class _FakeTypeSafe:
    def __init__(self, *, fail=False):
        self.fail = fail
        self.calls = []

    def system_one(self, *, state, questions):
        self.calls.append((state, questions))
        if self.fail:
            raise RuntimeError("upstream down")
        answers = {
            "verdict": _FakeAnswer(
                choice="flag",
                confidence=0.71,
                probabilities={"pass": 0.29, "flag": 0.71},
            )
        }
        for i, kind in enumerate(DEFECT_KINDS):
            answers[kind] = _FakeAnswer(noul=0.1 * (i + 1))
        return _FakeAnswer(
            answers=answers,
            model="jev-1.13.0",
            usage=_FakeAnswer(input_tokens=2500, output_tokens=0),
        )


def test_jev_runner_maps_choice_and_nouls_and_leaves_findings_blank(monkeypatch):
    pytest.importorskip(
        "typesafe_sdk",
        reason="typesafe-sdk not installed; run with --with typesafe-sdk==0.6.0",
    )
    fake = _FakeTypeSafe()
    runner = JevRunner("jev-1.13.0", client=fake)
    response = runner.judge(_case())
    assert response.verdict == "flag"
    assert response.verdict_score == 0.71
    assert response.kind_scores["amount_changed"] == pytest.approx(0.1)
    assert response.kind_scores["entity_wrong"] == pytest.approx(0.6)
    assert all(ch == CHANNEL_NATIVE for ch in response.kind_score_channels.values())
    assert response.findings == []
    assert response.tokens_input == 2500
    state, questions = fake.calls[0]
    assert set(questions) == {"verdict", *DEFECT_KINDS}
    assert state["provision_text_verbatim"] == truncate_provision(PROVISION, 24_000)
    assert runner.identity()["model"] == "jev-1.13.0"


def test_jev_runner_fails_closed_without_sdk_or_on_exception():
    pytest.importorskip("typesafe_sdk")
    runner = JevRunner("jev-1.13.0", client=_FakeTypeSafe(fail=True))
    response = runner.judge(_case())
    assert response.verdict == "error"
    assert response.error["type"] == "RuntimeError"


def test_jev_state_and_questions_cover_every_kind():
    assert set(KIND_NOULS) == set(DEFECT_KINDS)
    state = build_state(_case(), 24_000)
    assert set(state) == {
        "citation",
        "provision_text_verbatim",
        "generated_rulespec_artifact",
    }
    pytest.importorskip("typesafe_sdk")
    questions = build_questions()
    assert list(questions) == ["verdict", *DEFECT_KINDS]


def test_make_runner_parses_specs(tmp_path):
    replay = tmp_path / "r.json"
    replay.write_text("{}")
    assert isinstance(make_runner("replay:" + str(replay)), ReplayRunner)
    assert isinstance(make_runner("referee:claude-haiku-4-5-20251001"), RefereeRunner)
    assert isinstance(make_runner("jev"), JevRunner)
    with pytest.raises(ValueError):
        make_runner("referee")
    with pytest.raises(ValueError):
        make_runner("gpt:whatever")


def test_replay_runner_reports_missing_cases_as_errors(tmp_path):
    replay = tmp_path / "r.json"
    replay.write_text(
        json.dumps({"p1:defective": {"verdict": "flag", "verdict_score": 0.9}})
    )
    runner = ReplayRunner(replay)
    assert runner.judge(_case()).verdict == "flag"
    missing = runner.judge(_case(VARIANT_CONTROL))
    assert missing.verdict == "error" and missing.error["type"] == "missing_replay"


# -- results + board -------------------------------------------------------------


def _suite_for_board(seed=3, per_kind=2):
    suite, _ = build_synthetic_suite(
        _artifacts(14),
        name="board unit",
        source_kind="test",
        source_identity={},
        provision_chars=24_000,
        truncate=truncate_provision,
        per_kind=per_kind,
        seed=seed,
    )
    return suite


def _replay_file(tmp_path, suite, name, *, sharp=True, drop=(), error=()):
    rng = random.Random(0)
    responses = {}
    for case in suite.cases:
        if case.case_id in drop:
            continue
        if case.case_id in error:
            responses[case.case_id] = {
                "verdict": "error",
                "error": {"type": "boom", "message": "x"},
            }
            continue
        defective = case.is_defective
        if sharp:
            score = 0.9 if defective else 0.1
        else:
            score = 0.9  # flags everything
        kind_scores = {
            k: (0.95 if (defective and k == case.defect_kind) else 0.05)
            for k in DEFECT_KINDS
        }
        findings = []
        if defective:
            findings = [
                {
                    "kind": "amount_mismatch",
                    "rule_path": case.locator.rule_name,
                    "clause_ref": "c",
                    "explanation": "e",
                }
            ]
        responses[case.case_id] = {
            "verdict": "flag" if score > 0.5 else "pass",
            "verdict_score": score,
            "kind_scores": kind_scores,
            "kind_score_channels": {k: CHANNEL_NATIVE for k in DEFECT_KINDS},
            "findings": findings,
            "latency_ms": 100 + rng.randrange(50),
            "tokens": {"input": 1000, "output": 100},
        }
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps({"model": f"fake-{name}", "responses": responses}))
    return path


def test_run_suite_writes_contract_and_board_ranks_under_ceiling(tmp_path):
    suite = _suite_for_board()
    sharp = ReplayRunner(_replay_file(tmp_path, suite, "sharp"), name="sharp")
    noisy = ReplayRunner(
        _replay_file(tmp_path, suite, "noisy", sharp=False), name="noisy"
    )
    price = Price("fake-sharp", 1.0, 5.0, "unit")
    payload = run_suite(suite, sharp, tmp_path / "sharp", price=price, workers=2)
    run_suite(suite, noisy, tmp_path / "noisy", price=None, workers=2)
    assert payload["schema"] == RESULTS_SCHEMA
    assert payload["suite"]["sha256"] == suite.sha256
    assert payload["coverage"]["complete"] is True
    assert payload["results"][0]["cost_usd"] == pytest.approx(0.0015)
    loaded = load_results(tmp_path / "sharp")
    assert len(loaded["results"]) == len(suite.cases)

    board = board_module.fold_verifier_board(
        [tmp_path / "sharp", tmp_path / "noisy"], false_alarm_ceiling=0.25
    )
    ordered = board.ordered_runners()
    assert [s.runner for s in ordered] == ["sharp", "noisy"]
    assert ordered[0].over_ceiling is False
    assert ordered[0].native_false_alarm_rate == 0.0
    assert ordered[0].mean_kind_auc == 1.0
    assert ordered[0].localization_rate == 1.0
    assert ordered[0].total_cost_usd == pytest.approx(0.0015 * len(suite.cases))
    assert ordered[1].over_ceiling is True
    assert ordered[1].native_false_alarm_rate == 1.0
    assert ordered[1].mean_cost_usd is None
    for kind in DEFECT_KINDS:
        assert ordered[0].kinds[kind].kind_auc == 1.0
        assert ordered[0].kinds[kind].detection_at_ceiling == 1.0
        assert ordered[0].kinds[kind].complete_pairs == 2
    markdown = board_module.render_board_markdown(board)
    assert "false-alarm ceiling of 25%" in markdown
    assert "noisy†" in markdown
    text = board_module.render_board_text(board)
    assert "over ceiling, unranked" in text
    as_json = board_module.board_to_json(board)
    assert as_json["false_alarm_ceiling"] == 0.25
    assert as_json["runners"][0]["runner"] == "sharp"


def test_board_refuses_mismatched_suites_duplicate_runners_and_partial_runs(tmp_path):
    suite_a = _suite_for_board(seed=3)
    suite_b = _suite_for_board(seed=4)
    a = ReplayRunner(_replay_file(tmp_path, suite_a, "a"), name="a")
    b = ReplayRunner(_replay_file(tmp_path, suite_b, "b"), name="b")
    run_suite(suite_a, a, tmp_path / "a", price=None)
    run_suite(suite_b, b, tmp_path / "b", price=None)
    with pytest.raises(board_module.VerifierBoardError, match="not comparable"):
        board_module.fold_verifier_board([tmp_path / "a", tmp_path / "b"])
    a2 = ReplayRunner(_replay_file(tmp_path, suite_a, "a2"), name="a")
    run_suite(suite_a, a2, tmp_path / "a2", price=None)
    with pytest.raises(board_module.VerifierBoardError, match="two runs of one judge"):
        board_module.fold_verifier_board([tmp_path / "a", tmp_path / "a2"])
    # A run with errors is incomplete: refused unless --allow-partial.
    first = suite_a.cases[0].case_id
    err = ReplayRunner(
        _replay_file(tmp_path, suite_a, "err", error=(first,)), name="err"
    )
    payload = run_suite(suite_a, err, tmp_path / "err", price=None)
    assert (
        payload["coverage"]["complete"] is False and payload["coverage"]["errors"] == 1
    )
    with pytest.raises(board_module.VerifierBoardError, match="incomplete"):
        board_module.fold_verifier_board([tmp_path / "err"])
    board = board_module.fold_verifier_board([tmp_path / "err"], allow_partial=True)
    assert board.incomplete_sources
    assert board.runners[0].errors == 1


def test_results_loader_refuses_tampered_rows_and_wrong_schema(tmp_path):
    suite = _suite_for_board()
    runner = ReplayRunner(_replay_file(tmp_path, suite, "t"), name="t")
    run_suite(suite, runner, tmp_path / "t", price=None)
    path = tmp_path / "t" / "results.json"
    payload = json.loads(path.read_text())
    payload["results"][0]["verdict_score"] = 0.42
    path.write_text(json.dumps(payload))
    with pytest.raises(ResultsError, match="result_sha256"):
        load_results(path)
    payload = json.loads(path.read_text())
    payload["results"][0]["verdict_score"] = 0.9  # restore
    payload["results"][0]["result_sha256"] = canonical_json_sha256(
        {k: v for k, v in payload["results"][0].items() if k != "result_sha256"}
    )
    payload["schema"] = "something/else"
    path.write_text(json.dumps(payload))
    with pytest.raises(ResultsError, match="schema"):
        load_results(path)


def test_run_suite_resumes_without_rejudging_completed_cases(tmp_path):
    suite = _suite_for_board()
    replay = _replay_file(tmp_path, suite, "r")

    class Counting(ReplayRunner):
        calls = 0

        def judge(self, case):
            Counting.calls += 1
            return super().judge(case)

    runner = Counting(replay, name="r")
    run_suite(suite, runner, tmp_path / "r", price=None, limit=3)
    assert Counting.calls == 3
    run_suite(suite, runner, tmp_path / "r", price=None)
    assert Counting.calls == len(suite.cases)
    payload = load_results(tmp_path / "r")
    assert payload["coverage"]["complete"] is True


def test_error_rows_are_retried_on_resume(tmp_path):
    suite = _suite_for_board()
    first = suite.cases[0].case_id
    erroring = ReplayRunner(
        _replay_file(tmp_path, suite, "e1", error=(first,)), name="e"
    )
    run_suite(suite, erroring, tmp_path / "e", price=None)
    fixed = ReplayRunner(_replay_file(tmp_path, suite, "e2"), name="e")
    payload = run_suite(suite, fixed, tmp_path / "e", price=None)
    assert (
        payload["coverage"]["errors"] == 0 and payload["coverage"]["complete"] is True
    )


# -- CLI -------------------------------------------------------------------------


def test_cli_end_to_end_with_replay_runner(tmp_path, capsys):
    suite = _suite_for_board()
    suite.write(tmp_path / "suite")
    replay = _replay_file(tmp_path, suite, "cli")
    assert (
        verifier_cli.main(
            [
                "run",
                "--suite",
                str(tmp_path / "suite"),
                "--judge",
                f"replay:{replay}",
                "--name",
                "cli",
                "--out",
                str(tmp_path / "out"),
                "--quiet",
            ]
        )
        == 0
    )
    assert (
        verifier_cli.main(
            [
                "board",
                str(tmp_path / "out"),
                "--markdown-out",
                str(tmp_path / "board.md"),
                "--json-out",
                str(tmp_path / "board.json"),
                "--csv-out",
                str(tmp_path / "board.csv"),
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "False-alarm ceiling on native verdict: 25%" in out
    assert (
        (tmp_path / "board.md").read_text().startswith("# EncodeBench verifier board")
    )
    rows = (tmp_path / "board.csv").read_text().splitlines()
    assert len(rows) == 1 + len(DEFECT_KINDS)
    assert (
        verifier_cli.main(["board", str(tmp_path / "out"), str(tmp_path / "out")]) == 2
    )
    assert verifier_cli.main(["show-suite", str(tmp_path / "suite"), "--locators"]) == 0


def test_cli_build_real_from_fixture(tmp_path):
    assert (
        verifier_cli.main(
            ["build-real", "--dir", str(FIXTURE_REAL), "--out", str(tmp_path / "real")]
        )
        == 0
    )
    suite = CaseSuite.load(tmp_path / "real")
    assert suite.schema if hasattr(suite, "schema") else True
    assert (
        json.loads((tmp_path / "real" / "suite.json").read_text())["schema"]
        == SUITE_SCHEMA
    )


def test_judge_response_round_trip():
    response = JudgeResponse(
        verdict="flag",
        verdict_score=0.5,
        kind_scores={k: 0.1 for k in DEFECT_KINDS},
        kind_score_channels={k: CHANNEL_NATIVE for k in DEFECT_KINDS},
        findings=[],
        latency_ms=5,
        tokens_input=1,
        tokens_output=2,
        model="m",
    )
    assert JudgeResponse.from_dict(response.to_dict()) == response
    assert yaml.safe_load(dump_yaml_document({"a": "b\nc"})) == {"a": "b\nc"}


def test_localization_is_blank_for_probability_only_judges(tmp_path):
    from encodebench_verifier.results import result_row

    case = _case()
    response = JudgeResponse(
        verdict="flag",
        verdict_score=0.9,
        kind_scores={k: 0.5 for k in DEFECT_KINDS},
        kind_score_channels={k: CHANNEL_NATIVE for k in DEFECT_KINDS},
        findings=[],
        latency_ms=1,
        tokens_input=1,
        tokens_output=0,
        model="jev-1.13.0",
    )
    assert result_row(1, case, response, price=None)["localized"] is False
    blank = result_row(1, case, response, price=None, supports_localization=False)
    assert blank["localized"] is None and blank["localization_evidence"] is None
    pytest.importorskip("typesafe_sdk")
    assert (
        JevRunner("jev-1.13.0", client=_FakeTypeSafe()).supports_localization is False
    )
    assert RefereeRunner("claude-haiku-4-5-20251001").supports_localization is True
    replay = tmp_path / "r.json"
    replay.write_text(json.dumps({"responses": {}, "supports_localization": False}))
    assert ReplayRunner(replay).supports_localization is False
    # An error verdict never counts as a localisation miss either.
    errored = result_row(
        1,
        case,
        JudgeResponse.from_dict(
            {"verdict": "error", "error": {"type": "x", "message": "y"}}
        ),
        price=None,
    )
    assert errored["localized"] is None


def test_filtered_suite_records_parent_and_folds_existing_rows_without_rejudging(
    tmp_path,
):
    parent = _suite_for_board()
    replay = _replay_file(tmp_path, parent, "f")

    class Counting(ReplayRunner):
        calls = 0

        def judge(self, case):
            Counting.calls += 1
            return super().judge(case)

    runner = Counting(replay, name="f")
    run_suite(parent, runner, tmp_path / "f", price=None)
    judged = Counting.calls
    drop_citation = parent.cases[0].citation
    child = parent.filtered(
        name="child",
        keep_pair=lambda case: case.citation != drop_citation,
        description={"drop_citation": drop_citation},
    )
    assert child.sha256 != parent.sha256
    derived = child.source_identity["derived_from"]
    assert derived["parent_suite_sha256"] == parent.sha256
    assert derived["dropped_pairs"] == [parent.cases[0].pair_id]
    assert len(child.cases) == len(parent.cases) - 2
    assert all(c.citation != drop_citation for c in child.cases)
    # Re-assembling against the child re-judges nothing and re-stamps positions.
    payload = run_suite(child, runner, tmp_path / "f", price=None)
    assert Counting.calls == judged
    assert payload["suite"]["sha256"] == child.sha256
    assert payload["coverage"]["complete"] is True
    assert [row["index"] for row in payload["results"]] == list(
        range(1, len(child.cases) + 1)
    )
    loaded = load_results(tmp_path / "f")
    assert len(loaded["results"]) == len(child.cases)
    # Parent-suite and child-suite runs never fold together.
    other = ReplayRunner(_replay_file(tmp_path, parent, "g"), name="g")
    run_suite(parent, other, tmp_path / "g", price=None)
    with pytest.raises(board_module.VerifierBoardError, match="not comparable"):
        board_module.fold_verifier_board([tmp_path / "f", tmp_path / "g"])


def test_cli_filter_suite_by_citation_prefix(tmp_path):
    suite = _suite_for_board()
    suite.write(tmp_path / "suite")
    assert (
        verifier_cli.main(
            [
                "filter-suite",
                "--suite",
                str(tmp_path / "suite"),
                "--name",
                "x",
                "--out",
                str(tmp_path / "o"),
            ]
        )
        == 2
    )
    assert (
        verifier_cli.main(
            [
                "filter-suite",
                "--suite",
                str(tmp_path / "suite"),
                "--keep-citation-prefix",
                "us-de/",
                "--drop-citation-prefix",
                suite.cases[0].citation,
                "--name",
                "us only",
                "--out",
                str(tmp_path / "child"),
            ]
        )
        == 0
    )
    child = CaseSuite.load(tmp_path / "child")
    assert child.name == "us only"
    assert len(child.cases) == len(suite.cases) - 2


def test_runner_exception_is_recorded_as_error_row_not_a_crash(tmp_path):
    suite = _suite_for_board()

    class Exploding(ReplayRunner):
        def judge(self, case):
            if case.case_id == suite.cases[0].case_id:
                raise RuntimeError("sdk blew up")
            return super().judge(case)

    runner = Exploding(_replay_file(tmp_path, suite, "x"), name="x")
    payload = run_suite(suite, runner, tmp_path / "x", price=None)
    assert payload["coverage"]["errors"] == 1
    assert payload["coverage"]["complete"] is False
    row = payload["results"][0]
    assert row["verdict"] == "error"
    assert row["error"]["type"] == "runner_exception:RuntimeError"
    assert row["verdict_score"] is None and row["localized"] is None


def test_results_loader_refuses_case_identities_that_do_not_match_the_suite_digest(
    tmp_path,
):
    suite = _suite_for_board()
    runner = ReplayRunner(_replay_file(tmp_path, suite, "d"), name="d")
    run_suite(suite, runner, tmp_path / "d", price=None)
    path = tmp_path / "d" / "results.json"
    payload = json.loads(path.read_text())
    payload["case_identities"][0]["artifact_sha256"] = "0" * 64
    path.write_text(json.dumps(payload))
    with pytest.raises(ResultsError, match="suite sha256"):
        load_results(path)


def test_detection_at_ceiling_floor_is_float_safe():
    negatives = [i / 100 for i in range(100)]
    positives = [0.705, 0.715]
    # 0.29 * 100 must admit 29 controls (threshold 0.70), not 28 (threshold 0.71).
    assert detection_at_false_alarm_ceiling(positives, negatives, 0.29) == 1.0
    assert detection_at_false_alarm_ceiling(positives, negatives, 0.28) == 0.5


def test_eval_suite_source_reads_gate_pass_artifacts_through_the_board_loader(tmp_path):
    import hashlib

    from encodebench_verifier.sources import eval_suite as eval_suite_source

    from tests import test_eval_board as board_fixtures

    cases = board_fixtures.CASE_IDENTITIES
    rows = []
    for case in cases[:2]:
        row = board_fixtures._result("terra", case)
        artifact = tmp_path / "out" / "terra" / f"{case['index']}.yaml"
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_text(ARTIFACT)
        row["output_file"] = str(artifact)
        row["generated_output_sha256"] = hashlib.sha256(
            artifact.read_bytes()
        ).hexdigest()
        workspace = (
            tmp_path
            / "out"
            / "_eval_workspaces"
            / "terra"
            / eval_suite_source._slugify(case["corpus_citation_path"])
            / "workspace"
        )
        workspace.mkdir(parents=True)
        (workspace / "source.txt").write_text(PROVISION)
        rows.append(row)
    # A gate-failing row must be ignored even though its artifact is missing.
    failing = board_fixtures._result(
        "terra", cases[2], metrics=board_fixtures._metrics(ci_pass=False)
    )
    rows.append(failing)
    payload = board_fixtures._payload([("terra", "codex", "gpt-5.6-terra")], rows)
    (tmp_path / "out" / "results.json").write_text(json.dumps(payload))

    artifacts, identity = eval_suite_source.load_known_good([tmp_path / "out"])
    assert [a.citation for a in artifacts] == [
        c["corpus_citation_path"] for c in cases[:2]
    ]
    assert artifacts[0].provision_text == PROVISION
    assert artifacts[0].artifact_text == ARTIFACT
    assert artifacts[0].origin["generator_model"] == "gpt-5.6-terra"
    assert identity["corpus_releases"] == ["uk-rulespec-2026-07-14"]
    assert identity["suite_names"] == ["EncodeBench UK v1"]

    suite, report = build_synthetic_suite(
        artifacts,
        name="from eval suite",
        source_kind="eval_suite",
        source_identity=identity,
        provision_chars=24_000,
        truncate=truncate_provision,
        per_kind=1,
        seed=1,
        corpus_release=identity["corpus_releases"][0],
    )
    assert suite.corpus_release == "uk-rulespec-2026-07-14"
    assert suite.summary()["pair_count"] == 2

    # A tampered artifact is refused rather than mutated.
    (tmp_path / "out" / "terra" / "1.yaml").write_text(ARTIFACT + "# edited\n")
    with pytest.raises(
        eval_suite_source.EvalSuiteSourceError, match="generated_output_sha256"
    ):
        eval_suite_source.load_known_good([tmp_path / "out"])
