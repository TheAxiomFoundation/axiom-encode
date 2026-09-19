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
    assert MUTATOR_VERSION == "1.0.1"


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
    with pytest.raises(SuiteError, match="missing a control or defective"):
        CaseSuite(
            name="x",
            source_kind="t",
            source_identity={},
            cases=[case],
            provision_chars=10,
        )
    with pytest.raises(SuiteError, match="has no cases"):
        CaseSuite(
            name="x", source_kind="t", source_identity={}, cases=[], provision_chars=10
        )
    good = _suite_for_board()
    payload = good.to_dict()
    payload.pop("sha256")
    with pytest.raises(SuiteError, match="no sha256"):
        CaseSuite.from_dict(payload)
    assert CaseSuite.from_dict(payload, require_sha256=False).sha256 == good.sha256
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
    suite, report = real_source.build_real_suite(
        FIXTURE_REAL, provision_chars=24_000, truncate=truncate_provision
    )
    assert suite.source_kind == "real_defects"
    assert suite.summary()["pair_count"] == 2
    # rd-0003 is not a family representative, so the default selection drops
    # it before its missing artifacts would.
    assert report["skipped"] == {"family_member_not_representative": 1}
    assert report["records_seen"] == 3
    kinds = {c.pair_id: c.defect_kind for c in suite.cases if c.is_defective}
    assert kinds == {
        "rd-0001": "boundary_flipped",
        "rd-0002": "other:unrepresented_clause",
    }
    assert suite.source_identity["selection"]["representatives_only"] is True
    assert suite.corpus_release == "us-rulespec-2026-07-14"
    for case in suite.cases:
        assert case.control_clean == "unverified"
    defective = next(
        c for c in suite.cases if c.pair_id == "rd-0001" and c.is_defective
    )
    assert defective.locator.rule_name == "tax_table_method_applies"
    # The corpus schema carries rule names, not tokens: the rule name doubles
    # as the localisation token.
    assert defective.locator.token == "tax_table_method_applies"
    assert (
        defective.locator.path == "rules[tax_table_method_applies].versions[0].formula"
    )
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
    for key in (
        "pre_fix_artifact_sha256",
        "post_fix_artifact_sha256",
        "provision_sha256",
    ):
        payload.pop(key)
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
    suite, _ = real_source.build_real_suite(
        root, provision_chars=100, truncate=truncate_provision
    )
    defective = next(c for c in suite.cases if c.is_defective)
    assert defective.defect_kind == "other:wrong_import"
    assert defective.locator.rule_name == "some_rule"
    assert real_source.normalise_kind("Boundary Direction") == "boundary_flipped"
    assert real_source.normalise_kind("effective_date") == "date_or_period_wrong"
    assert (
        real_source.normalise_kind("wrong_period_or_effective_date")
        == "date_or_period_wrong"
    )
    assert real_source.normalise_kind("wrong_entity_or_scope") == "entity_wrong"
    assert real_source.normalise_kind("polarity_or_logic") == "polarity_swapped"
    assert (
        real_source.normalise_kind("untraceable_branch") == "other:untraceable_branch"
    )
    assert (
        real_source.normalise_kind("other", "a long free-text label") == "other:other"
    )


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
    assert verdict_score_from("flag", 7) == 1.0  # clamped
    assert verdict_score_from("pass", -1) == 1.0


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
    assert prices["claude-sonnet-4-5"].input_usd_per_million == 3.0
    assert "platform.claude.com" in prices["claude-sonnet-4-5"].source
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
    runner = RefereeRunner("claude-haiku-4-5-20251001", client_factory=lambda: fake)
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
        client_factory=lambda: _FakeJudgeClient(error="rate_limit"),
    )
    response = runner.judge(_case())
    assert response.verdict == "error"
    assert response.verdict_score is None
    assert response.error["type"] == "rate_limit"
    assert not response.ok


def test_referee_runner_pass_verdict_scores_low():
    fake = _FakeJudgeClient({"verdict": "pass", "confidence": 0.7, "findings": []})
    runner = RefereeRunner("claude-haiku-4-5-20251001", client_factory=lambda: fake)
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
    from encodebench_verifier.results import PAYLOAD_SHA256_FIELD, payload_sha256

    payload["results"][0]["verdict_score"] = 0.42
    path.write_text(json.dumps(payload))
    # The payload digest covers the rows, so any edit trips it first ...
    with pytest.raises(ResultsError, match="payload digest"):
        load_results(path)
    # ... and re-signing the envelope still leaves the row's own digest wrong.
    payload[PAYLOAD_SHA256_FIELD] = payload_sha256(payload)
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
    assert "False-alarm ceiling on native verdict: 10%" in out
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
    from encodebench_verifier.results import PAYLOAD_SHA256_FIELD, payload_sha256

    payload = json.loads(path.read_text())
    payload["case_identities"][0]["artifact_sha256"] = "0" * 64
    payload[PAYLOAD_SHA256_FIELD] = payload_sha256(payload)  # re-sign the envelope
    path.write_text(json.dumps(payload))
    with pytest.raises(ResultsError, match="suite sha256"):
        load_results(path)
    # Without re-signing, the envelope digest catches it first.
    payload["case_identities"][0]["case_id"] = "nope:control"
    path.write_text(json.dumps(payload))
    with pytest.raises(ResultsError, match="payload digest"):
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


def test_amount_guard_is_numeric_equality_not_substring():
    from decimal import Decimal

    from encodebench_verifier.mutator import provision_numbers

    artifact = ARTIFACT.replace("formula: '60000'", "formula: '200'")
    # "200" appears only inside "2008" and "3200(b)": not a stated amount.
    trap = "Beginning on September 1, 2008, under section 3200(b), a fee applies."
    assert mutate(artifact, trap, "amount_changed", rng=random.Random(1)) is None
    # Stated with separators and cents, it is the same number.
    stated = "The fee is $200.00 for each application, up to $60,000."
    mutation = mutate(artifact, stated, "amount_changed", rng=random.Random(1))
    assert mutation is not None and mutation.locator.before == "200"
    numbers = provision_numbers(
        "pay $1,250.50, or 20 percent, by 2026. Section 3211(b)."
    )
    assert Decimal("1250.5") in numbers and Decimal("20") in numbers
    assert Decimal("3211") in numbers and Decimal("11") not in numbers
    # The replacement must not be a stated number either.
    crowded = "Amounts of $200, $250, $300 and $160 are listed."
    mutation = mutate(artifact, crowded, "amount_changed", rng=random.Random(1))
    assert mutation is not None
    assert Decimal(mutation.locator.after) not in provision_numbers(crowded)


def test_year_guard_needs_the_year_as_its_own_word():
    # 2026 appears only inside a larger number and a form label.
    trap = "See account 120260 and form X2026; applies annually."
    for seed in range(10):
        mutation = mutate(
            ARTIFACT, trap, "date_or_period_wrong", rng=random.Random(seed)
        )
        assert mutation is None or mutation.locator.path.endswith(".period")


def test_conjunct_is_not_dropped_beside_a_top_level_or():
    document = load_yaml_document(ARTIFACT)
    document["rules"][1]["versions"][0]["formula"] = "a and b or c"
    document["rules"][2]["versions"][0]["formula"] = "x > 1"
    mixed = dump_yaml_document(document)
    assert (
        mutate(mixed, _window(PROVISION), "conjunct_dropped", rng=random.Random(1))
        is None
    )
    document["rules"][1]["versions"][0]["formula"] = "a and (b or c) and d"
    grouped = dump_yaml_document(document)
    mutation = mutate(
        grouped, _window(PROVISION), "conjunct_dropped", rng=random.Random(1)
    )
    assert mutation is not None
    formula = mutation.defective_document["rules"][1]["versions"][0]["formula"]
    assert formula in ("(b or c) and d", "a and d", "a and (b or c)")


def test_cli_filter_suite_drop_pair_requires_reason_and_known_ids(tmp_path):
    suite = _suite_for_board()
    suite.write(tmp_path / "suite")
    pair = suite.cases[0].pair_id
    base = ["filter-suite", "--suite", str(tmp_path / "suite"), "--name", "x"]
    base += ["--out", str(tmp_path / "o")]
    assert verifier_cli.main(base + ["--drop-pair", pair]) == 2
    assert verifier_cli.main(base + ["--drop-pair", "nope", "--reason", "r"]) == 2
    reason = "fails the 1.0.1 amount guard"
    assert verifier_cli.main(base + ["--drop-pair", pair, "--reason", reason]) == 0
    child = CaseSuite.load(tmp_path / "o")
    derived = child.source_identity["derived_from"]
    assert derived["dropped_pairs"] == [pair]
    assert derived["filter"]["reason"] == reason


# -- review round 1: contract and board fixes -----------------------------------


def test_resume_never_reuses_rows_from_another_judge_or_another_artifact(tmp_path):
    from encodebench_verifier.results import load_completed_rows

    suite = _suite_for_board()
    judge_a = ReplayRunner(_replay_file(tmp_path, suite, "a"), name="judge-a")
    run_suite(suite, judge_a, tmp_path / "shared", price=None)

    class Counting(ReplayRunner):
        calls = 0

        def judge(self, case):
            Counting.calls += 1
            return super().judge(case)

    # A different judge into the same --out re-judges everything.
    judge_b = Counting(_replay_file(tmp_path, suite, "b", sharp=False), name="judge-b")
    payload = run_suite(suite, judge_b, tmp_path / "shared", price=None)
    assert Counting.calls == len(suite.cases)
    assert payload["runner"]["name"] == "judge-b"
    assert all(row["runner_name"] == "judge-b" for row in payload["results"])
    assert all(row["model"] == "fake-b" for row in payload["results"])
    # A rebuilt suite with the same case ids but different artifact text
    # (different seed) re-judges too: rows are bound to content digests.
    other = _suite_for_board(seed=4)
    shared_ids = {c.case_id for c in suite.cases} & {c.case_id for c in other.cases}
    assert shared_ids, "fixture suites must share case ids for this test"
    Counting.calls = 0
    judge_b2 = Counting(
        _replay_file(tmp_path, other, "b2", sharp=False), name="judge-b"
    )
    run_suite(other, judge_b2, tmp_path / "shared", price=None)
    reused = len(
        load_completed_rows(
            tmp_path / "shared", suite=other, runner_identity_digest=None
        )
    )
    assert Counting.calls >= len(other.cases) - reused
    assert Counting.calls > 0


def test_results_loader_refuses_rows_from_a_different_runner_identity(tmp_path):
    from encodebench_verifier.results import PAYLOAD_SHA256_FIELD, payload_sha256

    suite = _suite_for_board()
    runner = ReplayRunner(_replay_file(tmp_path, suite, "r"), name="r")
    run_suite(suite, runner, tmp_path / "r", price=None)
    path = tmp_path / "r" / "results.json"
    payload = json.loads(path.read_text())
    payload["runner"]["name"] = "renamed"
    payload["runner"]["identity_sha256"] = canonical_json_sha256(
        {
            "name": "renamed",
            "family": payload["runner"]["family"],
            "model": payload["runner"]["model"],
            "identity": payload["runner"]["identity"],
        }
    )
    payload[PAYLOAD_SHA256_FIELD] = payload_sha256(payload)
    path.write_text(json.dumps(payload))
    with pytest.raises(ResultsError, match="different runner"):
        load_results(path)


def test_board_refuses_the_same_judge_identity_under_two_names(tmp_path):
    suite = _suite_for_board()
    replay = _replay_file(tmp_path, suite, "same")
    run_suite(suite, ReplayRunner(replay, name="one"), tmp_path / "one", price=None)
    run_suite(suite, ReplayRunner(replay, name="two"), tmp_path / "two", price=None)
    with pytest.raises(board_module.VerifierBoardError, match="same judge identity"):
        board_module.fold_verifier_board([tmp_path / "one", tmp_path / "two"])


def test_board_marks_a_judge_with_no_scored_controls_unrankable(tmp_path):
    suite = _suite_for_board()
    controls = tuple(c.case_id for c in suite.cases if not c.is_defective)
    cheat = ReplayRunner(
        _replay_file(tmp_path, suite, "cheat", error=controls), name="cheat"
    )
    honest = ReplayRunner(
        _replay_file(tmp_path, suite, "honest", sharp=False), name="honest"
    )
    run_suite(suite, cheat, tmp_path / "cheat", price=None)
    run_suite(suite, honest, tmp_path / "honest", price=None)
    board = board_module.fold_verifier_board(
        [tmp_path / "cheat", tmp_path / "honest"], allow_partial=True
    )
    by_name = {s.runner: s for s in board.runners}
    assert by_name["cheat"].rank_status == "unrankable"
    assert by_name["cheat"].native_false_alarm_rate is None
    assert by_name["honest"].rank_status == "over_ceiling"
    # Unrankable sorts last, behind over-ceiling judges.
    assert [s.runner for s in board.ordered_runners()] == ["honest", "cheat"]
    markdown = board_module.render_board_markdown(board)
    assert "cheat§" in markdown and "honest†" in markdown
    assert any("Unrankable" in note for note in board.notes)


def test_board_spend_and_latency_cover_error_rows_and_blank_unknown_usage(tmp_path):
    suite = _suite_for_board()
    first = suite.cases[0].case_id
    responses = json.loads(_replay_file(tmp_path, suite, "sp").read_text())
    responses["responses"][first] = {
        "verdict": "error",
        "error": {"type": "timeout", "message": "slow"},
        "latency_ms": 60_000,
        "tokens": {"input": 5000, "output": 0},
    }
    # One scored row with no usage reported at all.
    second = suite.cases[1].case_id
    responses["responses"][second].pop("tokens")
    path = tmp_path / "sp.json"
    path.write_text(json.dumps(responses))
    price = Price("fake-sp", 1.0, 5.0, "unit")
    payload = run_suite(
        suite, ReplayRunner(path, name="sp"), tmp_path / "sp", price=price
    )
    rows = {r["case_id"]: r for r in payload["results"]}
    assert rows[first]["cost_usd"] == pytest.approx(0.005)
    assert rows[second]["tokens"] == {"input": None, "output": None}
    assert rows[second]["cost_usd"] is None
    board = board_module.fold_verifier_board([tmp_path / "sp"], allow_partial=True)
    stats = board.runners[0]
    n = len(suite.cases)
    # Error row's cost and latency are in the totals; the unpriced row is counted.
    assert stats.total_cost_usd == pytest.approx(0.005 + 0.0015 * (n - 2))
    assert stats.unpriced_rows == 1
    assert stats.median_latency_seconds is not None
    assert stats.cases_scored == n - 1


def test_board_handles_other_kinds_and_unverified_controls(tmp_path):
    suite, _ = real_source.build_real_suite(
        FIXTURE_REAL, provision_chars=24_000, truncate=truncate_provision
    )
    # Relabel one pair as an out-of-taxonomy kind through a hand-built suite.
    cases = []
    for case in suite.cases:
        kind = "other:wrong_import" if case.pair_id == "rd-0002" else case.defect_kind
        cases.append(
            VerifierCase(
                pair_id=case.pair_id,
                variant=case.variant,
                defect_kind=kind,
                citation=case.citation,
                provision_text=case.provision_text,
                artifact_text=case.artifact_text,
                locator=case.locator,
                origin=case.origin,
                control_clean=case.control_clean,
            )
        )
    mixed = CaseSuite(
        name="real mixed",
        source_kind="real_defects",
        source_identity={},
        cases=cases,
        provision_chars=24_000,
    )
    flag_all = ReplayRunner(_replay_file(tmp_path, mixed, "fa", sharp=False), name="fa")
    run_suite(mixed, flag_all, tmp_path / "fa", price=None)
    board = board_module.fold_verifier_board([tmp_path / "fa"])
    assert board.ceiling_applied is False
    assert board.kinds == ["boundary_flipped", "other:wrong_import"]
    assert board.ceiling_applied is False
    stats = board.runners[0]
    # Flagging unverified post-fix controls does not unrank a judge.
    assert stats.native_false_alarm_rate == 1.0
    assert stats.rank_status == "ranked"
    assert stats.kinds["other:wrong_import"].channel == CHANNEL_VERDICT_FALLBACK
    assert stats.kinds["other:wrong_import"].pairs == 1
    markdown = board_module.render_board_markdown(board)
    assert "not proven clean" in markdown
    assert "other:wrong_import" in markdown
    assert "ceiling not applied" in markdown


def test_board_mean_auc_requires_every_kind_and_marks_fallbacks(tmp_path):
    suite = _suite_for_board()
    responses = json.loads(_replay_file(tmp_path, suite, "fb").read_text())
    for case_id, entry in responses["responses"].items():
        entry["kind_score_channels"]["entity_wrong"] = CHANNEL_VERDICT_FALLBACK
        entry["kind_scores"]["entity_wrong"] = entry["verdict_score"]
    path = tmp_path / "fb.json"
    path.write_text(json.dumps(responses))
    run_suite(suite, ReplayRunner(path, name="fb"), tmp_path / "fb", price=None)
    board = board_module.fold_verifier_board([tmp_path / "fb"])
    stats = board.runners[0]
    assert stats.verdict_fallback_kinds == ["entity_wrong"]
    assert stats.mean_kind_auc is not None and stats.mean_native_kind_auc is not None
    markdown = board_module.render_board_markdown(board)
    assert "1.000‡ |" in markdown  # the mean cell carries the fallback mark
    # A judge missing a whole kind's AUC has no rankable mean.
    entity = tuple(c.case_id for c in suite.cases if c.defect_kind == "entity_wrong")
    partial = ReplayRunner(_replay_file(tmp_path, suite, "pt", error=entity), name="pt")
    run_suite(suite, partial, tmp_path / "pt", price=None)
    board = board_module.fold_verifier_board([tmp_path / "pt"], allow_partial=True)
    assert board.runners[0].mean_kind_auc is None
    assert board.runners[0].missing_kind_aucs == ["entity_wrong"]
    assert board.runners[0].rank_status == "unrankable"


def test_derived_suite_provenance_is_digest_bound_and_shown_on_the_board(tmp_path):
    parent = _suite_for_board()
    drop = parent.cases[0].citation
    child = parent.filtered(
        name="child", keep_pair=lambda c: c.citation != drop, description={"drop": drop}
    )
    assert child.identity()["derived_from"]["parent_suite_sha256"] == parent.sha256
    child.write(tmp_path / "child")
    payload = json.loads((tmp_path / "child" / "suite.json").read_text())
    payload["source"]["identity"]["derived_from"]["dropped_pairs"] = []
    (tmp_path / "child" / "suite.json").write_text(json.dumps(payload))
    with pytest.raises(SuiteError, match="sha256"):
        CaseSuite.load(tmp_path / "child")
    runner = ReplayRunner(_replay_file(tmp_path, child, "c"), name="c")
    run_suite(child, runner, tmp_path / "c", price=None)
    board = board_module.fold_verifier_board([tmp_path / "c"])
    assert any("Derived suite" in note for note in board.notes)
    with pytest.raises(SuiteError, match="has no cases"):
        parent.filtered(name="empty", keep_pair=lambda c: False, description={})


def test_cases_jsonl_tolerates_one_truncated_tail_and_unicode_separators(tmp_path):
    from encodebench_verifier.results import read_rows

    suite = _suite_for_board()
    responses = json.loads(_replay_file(tmp_path, suite, "u").read_text())
    for entry in responses["responses"].values():
        entry["findings"] = [
            {
                "kind": "amount_mismatch",
                "rule_path": "r",
                "clause_ref": "c",
                "explanation": "line\u2028separator and \u0085 next line",
            }
        ]
    path = tmp_path / "u.json"
    path.write_text(json.dumps(responses))
    run_suite(suite, ReplayRunner(path, name="u"), tmp_path / "u", price=None)
    jsonl = tmp_path / "u" / "cases.jsonl"
    rows = read_rows(jsonl)
    assert len(rows) == len(suite.cases)
    assert "\u2028" in rows[0]["findings"][0]["explanation"]
    # A crash mid-write leaves a partial last line: dropped with a warning.
    text = jsonl.read_text()
    jsonl.write_text(text + '{"case_id": "half')
    with pytest.warns(UserWarning, match="truncated"):
        assert len(read_rows(jsonl)) == len(suite.cases)
    # Interior corruption is still refused.
    jsonl.write_text('{"broken\n' + text)
    with pytest.raises(ResultsError, match="not JSON"):
        read_rows(jsonl)


def test_fresh_rotates_instead_of_deleting_and_limit_keeps_finished_rows(tmp_path):
    suite = _suite_for_board()
    runner = ReplayRunner(_replay_file(tmp_path, suite, "f"), name="f")
    full = run_suite(suite, runner, tmp_path / "f", price=None)
    assert full["coverage"]["complete"] is True
    # A spot check with --limit never downgrades a finished run.
    spot = run_suite(suite, runner, tmp_path / "f", price=None, limit=2)
    assert spot["coverage"]["complete"] is True
    with pytest.raises(ValueError, match="at least 1"):
        run_suite(suite, runner, tmp_path / "f", price=None, limit=0)
    run_suite(suite, runner, tmp_path / "f", price=None, resume=False, limit=1)
    backups = sorted(p.name for p in (tmp_path / "f").glob("*.bak"))
    assert any(name.startswith("cases.jsonl.") for name in backups)
    assert any(name.startswith("results.json.") for name in backups)
    assert load_results(tmp_path / "f")["coverage"]["complete"] is False


def test_interrupt_cancels_queued_cases_and_keeps_finished_rows(tmp_path):
    suite = _suite_for_board()
    replay = _replay_file(tmp_path, suite, "i")

    class Interrupting(ReplayRunner):
        calls = 0

        def judge(self, case):
            Interrupting.calls += 1
            if Interrupting.calls == 3:
                raise KeyboardInterrupt
            return super().judge(case)

    runner = Interrupting(replay, name="i")
    with pytest.raises(KeyboardInterrupt):
        run_suite(suite, runner, tmp_path / "i", price=None, workers=1)
    assert Interrupting.calls < len(suite.cases)
    partial = json.loads((tmp_path / "i" / "results.json").read_text())
    assert partial["coverage"]["complete"] is False
    assert 0 < len(partial["results"]) < len(suite.cases)


def test_replay_runner_fails_closed_on_schema_drift(tmp_path):
    replay = tmp_path / "drift.json"
    replay.write_text(
        json.dumps(
            {
                "p1:defective": {"verdict_score": 0.1},
                "p1:control": {"verdict": "PASS"},
                "p2:control": {"verdict": "flag", "verdict_score": -3},
                "p2:defective": {
                    "verdict": "flag",
                    "verdict_score": 0.9,
                    "error": {"type": "x", "message": "y"},
                },
                "p3:defective": {"verdict": "flag", "verdict_score": 0.9},
            }
        )
    )
    runner = ReplayRunner(replay)
    for pair, variant in (
        ("p1", "defective"),
        ("p1", "control"),
        ("p2", "control"),
        ("p2", "defective"),
    ):
        case = (
            _case(variant) if variant == VARIANT_DEFECTIVE else _case(VARIANT_CONTROL)
        )
        object.__setattr__(case, "pair_id", pair)
        response = runner.judge(case)
        assert response.verdict == "error", (pair, variant, response)
        assert response.error["type"] in ("invalid_response",)
    # A verdict-score-only recording is completed with verdict fallbacks.
    case = _case(VARIANT_DEFECTIVE)
    object.__setattr__(case, "pair_id", "p3")
    response = runner.judge(case)
    assert response.ok
    assert all(v == 0.9 for v in response.kind_scores.values())
    assert set(response.kind_score_channels.values()) == {CHANNEL_VERDICT_FALLBACK}


def test_referee_verdict_matches_production_run_for_every_payload_shape():
    from axiom_encode.judges.run_log import Verdict

    payloads = [
        {"verdict": "pass", "confidence": 0.8, "findings": []},
        {"verdict": "flag", "confidence": 0.8, "findings": []},
        {
            "verdict": "pass",
            "confidence": 0.9,
            "findings": [
                {
                    "clause_ref": "c",
                    "rule_path": "r",
                    "kind": "amount_mismatch",
                    "explanation": "e",
                }
            ],
        },
        {
            "verdict": "flag",
            "confidence": 0.6,
            "findings": [
                {
                    "clause_ref": "c",
                    "rule_path": "r",
                    "kind": "typo",
                    "explanation": "e",
                }
            ],
        },
        {"verdict": "maybe", "confidence": 0.5, "findings": []},
    ]
    for payload in payloads:
        fake = _FakeJudgeClient(payload)
        runner = RefereeRunner("claude-haiku-4-5-20251001", client_factory=lambda: fake)
        ours = runner.judge(_case())
        production = statutory_fidelity.run(
            PROVISION, ARTIFACT, citation="us-de/statute/30/1102", client=fake
        )
        expected = {Verdict.PASS: "pass", Verdict.FLAG: "flag", Verdict.ERROR: "error"}[
            production.verdict
        ]
        assert ours.verdict == expected, payload
        if ours.ok:
            assert len(ours.findings) == len(production.findings)
    # The coerced pass-with-findings case is recorded, and scored on the
    # production verdict like the calibration harness does.
    fake = _FakeJudgeClient(payloads[2])
    response = RefereeRunner(
        "claude-haiku-4-5-20251001", client_factory=lambda: fake
    ).judge(_case())
    assert response.verdict == "flag" and response.raw["raw_verdict"] == "pass"
    assert response.raw["verdict_coerced_by_findings"] is True
    assert response.verdict_score == 0.9
    # An unknown finding kind is recorded but never credited as a native hit.
    fake = _FakeJudgeClient(payloads[3])
    response = RefereeRunner(
        "claude-haiku-4-5-20251001", client_factory=lambda: fake
    ).judge(_case())
    assert response.raw["unknown_finding_kinds"] == ["typo"]
    assert response.kind_scores["polarity_swapped"] == 0.0
    # An unparseable confidence is an error, not a sharp score.
    fake = _FakeJudgeClient({"verdict": "pass", "confidence": "high", "findings": []})
    response = RefereeRunner(
        "claude-haiku-4-5-20251001", client_factory=lambda: fake
    ).judge(_case())
    assert (
        response.verdict == "error"
        and response.error["type"] == "confidence_unparseable"
    )

    # A raising client factory (no SDK, no key) is an error row, not a crash.
    def boom():
        raise RuntimeError("no key")

    response = RefereeRunner("claude-haiku-4-5-20251001", client_factory=boom).judge(
        _case()
    )
    assert response.verdict == "error" and response.error["type"] == "RuntimeError"


def test_referee_records_generator_and_same_family_instead_of_refusing():
    fake = _FakeJudgeClient({"verdict": "pass", "confidence": 0.7, "findings": []})
    runner = RefereeRunner("claude-haiku-4-5-20251001", client_factory=lambda: fake)
    case = _case()
    claude_case = (
        VerifierCase(
            **{
                **case.to_dict_for_test(),
                "origin": {"generator_model": "claude-opus-4-6"},
            }
        )
        if hasattr(case, "to_dict_for_test")
        else None
    )
    if claude_case is None:
        claude_case = VerifierCase(
            pair_id=case.pair_id,
            variant=case.variant,
            defect_kind=case.defect_kind,
            citation=case.citation,
            provision_text=case.provision_text,
            artifact_text=case.artifact_text,
            locator=case.locator,
            origin={"generator_model": "claude-opus-4-6"},
        )
    response = runner.judge(claude_case)
    assert response.ok
    assert response.raw["same_family_as_generator"] is True
    unknown = VerifierCase(
        pair_id=case.pair_id,
        variant=case.variant,
        defect_kind=case.defect_kind,
        citation=case.citation,
        provision_text=case.provision_text,
        artifact_text=case.artifact_text,
        locator=case.locator,
        origin={},
    )
    response = runner.judge(unknown)
    assert response.ok and response.raw["same_family_as_generator"] is None
    assert runner.identity()["max_tokens"] == 2048


def test_jev_runner_fail_closed_variants():
    pytest.importorskip("typesafe_sdk")

    def make(answers=None, model="jev-1.13.0", usage=(2500, 0)):
        class Client:
            def system_one(self, *, state, questions):
                a = (
                    answers
                    if answers is not None
                    else {
                        "verdict": _FakeAnswer(
                            choice="flag",
                            confidence=0.7,
                            probabilities={"pass": 0.3, "flag": 0.7},
                        ),
                        **{k: _FakeAnswer(noul=0.5) for k in DEFECT_KINDS},
                    }
                )
                return _FakeAnswer(
                    answers=a,
                    model=model,
                    usage=_FakeAnswer(input_tokens=usage[0], output_tokens=usage[1]),
                )

        return Client()

    base = {
        "verdict": _FakeAnswer(
            choice="flag", confidence=0.7, probabilities={"pass": 0.3, "flag": 0.7}
        ),
        **{k: _FakeAnswer(noul=0.5) for k in DEFECT_KINDS},
    }
    # Missing Noul -> error, never a partial score.
    missing = dict(base)
    missing.pop("date_or_period_wrong")
    r = JevRunner("jev-1.13.0", client=make(missing)).judge(_case())
    assert r.verdict == "error" and "date_or_period_wrong" in r.error["message"]
    # Served model other than the pinned one -> error; alias accepted.
    r = JevRunner("jev-1.13.0", client=make(model="jev-1.14.0")).judge(_case())
    assert r.verdict == "error" and r.error["type"] == "served_model_mismatch"
    r = JevRunner("jev-latest", client=make(model="jev-1.14.0")).judge(_case())
    assert r.ok and r.model == "jev-1.14.0"
    # Unrecognised choice -> error without scores.
    odd = dict(base)
    odd["verdict"] = _FakeAnswer(
        choice="unsure", confidence=0.5, probabilities={"pass": 0.5, "flag": 0.5}
    )
    r = JevRunner("jev-1.13.0", client=make(odd)).judge(_case())
    assert r.verdict == "error" and r.verdict_score is None
    assert all(v is None for v in r.kind_scores.values())
    # Unreported usage stays None, and costs nothing rather than $0.
    r = JevRunner("jev-1.13.0", client=make(usage=(None, None))).judge(_case())
    assert r.ok and r.tokens_input is None and r.tokens_output is None
    assert (
        cost_usd(Price("jev-1.13.0", 0.042, 0.0, "s"), r.tokens_input, r.tokens_output)
        is None
    )

    # A client constructor that raises (no key) is an error row.
    class NoKey:
        def __init__(self):
            raise RuntimeError("No API key was provided")

    runner = JevRunner("jev-1.13.0")
    runner._get_client = lambda: NoKey()  # type: ignore[assignment]
    r = runner.judge(_case())
    assert r.verdict == "error" and r.error["type"] == "RuntimeError"


def test_pricing_loader_validates_entries(tmp_path):
    bad = tmp_path / "p.json"
    bad.write_text(json.dumps({"models": {"m": {"input_usd_per_million": 1.0}}}))
    with pytest.raises(ValueError, match="source"):
        load_pricing(bad)
    bad.write_text(
        json.dumps(
            {
                "models": {
                    "m": {
                        "input_usd_per_million": -1,
                        "output_usd_per_million": 0,
                        "source": "s",
                    }
                }
            }
        )
    )
    with pytest.raises(ValueError, match="negative"):
        load_pricing(bad)
    bad.write_text("not json")
    with pytest.raises(ValueError, match="could not read"):
        load_pricing(bad)


def test_cli_exit_codes_for_bad_inputs_and_retry_budget_mapping(tmp_path, monkeypatch):
    # A missing suite path is a usage error (2), not a traceback.
    assert verifier_cli.main(["show-suite", str(tmp_path / "nope")]) == 2
    assert (
        verifier_cli.main(
            [
                "run",
                "--suite",
                str(tmp_path / "nope"),
                "--judge",
                "jev",
                "--out",
                str(tmp_path / "o"),
            ]
        )
        == 2
    )
    # --max-attempts N gives Jev N-1 retries after the first attempt.
    captured = {}

    def fake_make_runner(spec, **options):
        captured.update(options)
        raise ValueError("stop here")

    suite = _suite_for_board()
    suite.write(tmp_path / "suite")
    monkeypatch.setattr(verifier_cli, "make_runner", fake_make_runner)
    assert (
        verifier_cli.main(
            [
                "run",
                "--suite",
                str(tmp_path / "suite"),
                "--judge",
                "jev",
                "--out",
                str(tmp_path / "o"),
                "--max-attempts",
                "1",
            ]
        )
        == 2
    )
    assert captured["max_retries"] == 0 and captured["max_attempts"] == 1


def test_agreement_joins_on_identical_text_and_refuses_different_judges(tmp_path):
    from encodebench_verifier.agreement import AgreementError, compare_runs

    # Distinct artifact text per pair, so content digests do not collide.
    distinct = [
        KnownGoodArtifact(
            key=a.key,
            citation=a.citation,
            provision_text=a.provision_text,
            artifact_text=a.artifact_text.replace("Two lines.", f"Variant {i}."),
            origin=a.origin,
        )
        for i, a in enumerate(_artifacts(14))
    ]
    suite, _ = build_synthetic_suite(
        distinct,
        name="agreement unit",
        source_kind="test",
        source_identity={},
        provision_chars=24_000,
        truncate=truncate_provision,
        per_kind=2,
        seed=3,
    )
    first = ReplayRunner(_replay_file(tmp_path, suite, "t1"), name="t1")
    run_suite(suite, first, tmp_path / "t1", price=None)
    # Second run: same judge, verdict flipped on one case, findings changed on another.
    responses = json.loads((tmp_path / "t1.json").read_text())
    flipped, changed = suite.cases[0].case_id, suite.cases[1].case_id
    responses["responses"][flipped]["verdict"] = (
        "pass" if responses["responses"][flipped]["verdict"] == "flag" else "flag"
    )
    responses["responses"][flipped]["verdict_score"] = 0.5
    responses["responses"][changed]["findings"] = [
        {
            "kind": "boundary_direction",
            "rule_path": "x",
            "clause_ref": "c",
            "explanation": "e",
        }
    ]
    path = tmp_path / "t2.json"
    path.write_text(json.dumps(responses))
    run_suite(suite, ReplayRunner(path, name="t2"), tmp_path / "t2", price=None)
    report = compare_runs(tmp_path / "t1", tmp_path / "t2")
    n = len(suite.cases)
    assert report.joined == n
    assert report.verdict_agreements == n - 1
    assert report.same_identity is False  # the response file name is part of identity
    assert report.max_abs_verdict_score_delta == pytest.approx(0.4)
    assert "identical texts judged in both runs" in report.render()
    # A different model is not a self-agreement comparison.
    other = json.loads(path.read_text())
    other["model"] = "another-model"
    (tmp_path / "t3.json").write_text(json.dumps(other))
    run_suite(
        suite,
        ReplayRunner(tmp_path / "t3.json", name="t3"),
        tmp_path / "t3",
        price=None,
    )
    with pytest.raises(AgreementError, match="one judge"):
        compare_runs(tmp_path / "t1", tmp_path / "t3")
    assert (
        verifier_cli.main(["agreement", str(tmp_path / "t1"), str(tmp_path / "t2")])
        == 0
    )
    assert (
        verifier_cli.main(["agreement", str(tmp_path / "t1"), str(tmp_path / "t3")])
        == 2
    )


def test_real_loader_selection_filters_and_reads_readme_keys(tmp_path):
    suite, report = real_source.build_real_suite(
        FIXTURE_REAL,
        provision_chars=24_000,
        truncate=truncate_provision,
        representatives_only=False,
        triage_statuses=(),
    )
    # rd-0003 is a family member and metadata-only: kept by the filters but skipped
    # at load because it ships no artifacts.
    assert report["skipped"] == {"metadata_only": 1}
    assert suite.summary()["pair_count"] == 2
    _, report = real_source.build_real_suite(
        FIXTURE_REAL,
        provision_chars=24_000,
        truncate=truncate_provision,
        min_confidence=0.85,
    )
    assert report["pairs_kept"] == 1 and report["skipped"]["below_min_confidence"] == 1
    with pytest.raises(real_source.RealDefectsError, match="has no cases"):
        real_source.build_real_suite(
            FIXTURE_REAL,
            provision_chars=24_000,
            truncate=truncate_provision,
            jurisdictions=("uk",),
        )
    defective = next(
        c for c in suite.cases if c.pair_id == "rd-0001" and c.is_defective
    )
    assert (
        defective.locator.path == "rules[tax_table_method_applies].versions[0].formula"
    )
    assert defective.locator.rule_name == "tax_table_method_applies"
    assert "pre-fix lines" in defective.locator.detail
    assert defective.origin["fix_reference"].endswith("/pull/1")
    assert defective.origin["fix_stage"] == "post_merge"
    assert (
        verifier_cli.main(
            [
                "build-real",
                "--dir",
                str(FIXTURE_REAL),
                "--out",
                str(tmp_path / "r"),
                "--jurisdiction",
                "us",
                "--min-confidence",
                "0.5",
            ]
        )
        == 0
    )
    assert (
        verifier_cli.main(
            [
                "build-real",
                "--dir",
                str(FIXTURE_REAL),
                "--out",
                str(tmp_path / "r2"),
                "--jurisdiction",
                "uk",
            ]
        )
        == 2
    )


def test_cli_filter_suite_by_max_case_chars(tmp_path):
    suite = _suite_for_board()
    suite.write(tmp_path / "suite")
    biggest = max(len(c.provision_text) + len(c.artifact_text) for c in suite.cases)
    base = ["filter-suite", "--suite", str(tmp_path / "suite"), "--name", "small"]
    assert (
        verifier_cli.main(
            base + ["--out", str(tmp_path / "o"), "--max-case-chars", "0"]
        )
        == 2
    )
    # A limit below every case leaves nothing: refused, not an empty suite.
    assert (
        verifier_cli.main(
            base + ["--out", str(tmp_path / "o"), "--max-case-chars", "10"]
        )
        == 2
    )
    assert (
        verifier_cli.main(
            base + ["--out", str(tmp_path / "o"), "--max-case-chars", str(biggest)]
        )
        == 0
    )
    child = CaseSuite.load(tmp_path / "o")
    assert len(child.cases) == len(suite.cases)
    assert child.source_identity["derived_from"]["filter"]["max_case_chars"] == biggest
