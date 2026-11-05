"""Historical actual captures; synthetic closure IDs are not authentication."""

import json
from pathlib import Path

import pytest

from axiom_encode.harness.same_case_execution import (
    CaseExecution,
    ClosureFile,
    SameCaseEvidence,
)
from axiom_encode.harness.student_loan_schedule_binding import (
    bind_student_loan_schedule,
)
from axiom_encode.harness.student_loan_schedule_execution import (
    observe_student_loan_schedule,
)
from axiom_encode.student_loan_schedule_source import recognize_student_loan_schedule

ROOT = "ca/policies/revenu-quebec/tp1-2025/schedule-m-student-loan-interest.yaml"
CITATION = "ca/policy/revenu-quebec/tp1-2025/schedule-m-student-loan-interest"
FIXTURE = Path(__file__).parent / "fixtures/student_loan_schedule/observed-graph.json"


def data():
    return json.loads(FIXTURE.read_text())


def evidence(tmp_path, d, *, stale=False):
    original = tmp_path / "original" / ROOT
    original.parent.mkdir(parents=True)
    original.write_text(d["content"] + ("\n" if stale else ""))
    snapshot = tmp_path / "snapshot"
    target = snapshot / ROOT
    target.parent.mkdir(parents=True)
    target.write_text(d["content"])
    return SameCaseEvidence(
        (ClosureFile(ROOT, original, d["content"].encode()),),
        ROOT,
        snapshot,
        "a" * 40,
        "b" * 64,
        d.get("engine_pin", "89571cc2a938707fd60a5489b951135345697296"),
        json.dumps(d["artifact"]).encode(),
        tuple(
            CaseExecution(
                r["case_sha256"],
                json.dumps(r["request"]),
                json.dumps(r["response"]),
                r["case_index"],
            )
            for r in d["records"]
        ),
    )


def observe(tmp_path, d, *, stale=False):
    captured = evidence(tmp_path, d, stale=stale)
    source = recognize_student_loan_schedule(d["source"], CITATION)
    binding = bind_student_loan_schedule(source, d["content"])
    assert binding is not None
    return observe_student_loan_schedule(binding, captured, d["content"], d["cases"])


def test_actual18_cases48_outputs_and_distinct_owner_pairs(tmp_path):
    d = data()
    assert len(d["records"]) == 18
    assert sum(len(c["output"]) for c in d["cases"]) == 48
    result = observe(tmp_path, d)
    assert result is not None
    assert len(result.duty_pairs) == 5
    assert result.equal_provider_pair and len(set(result.exclusive_provider_cases)) == 2


@pytest.mark.parametrize(
    "change",
    [
        "entity",
        "entity_id",
        "bool_int",
        "foreign_fact",
        "value",
        "frame",
        "fallback",
        "extra_metadata",
        "response_value",
        "expression",
        "missing_trace",
        "duplicate_fact",
        "compiled_branch",
        "compiled_bool",
        "version_bool",
        "cad_float",
        "cad_duplicate",
        "extra_rule",
        "missing_record",
        "changed_case",
        "missing_guide_pair",
        "missing_historical_provider",
        "missing_equal_provider",
        "missing_upper_boundary",
        "stale",
    ],
)
def test_actual_or_compiled_tampering_refuses(tmp_path, change):
    d = data()
    request = d["records"][0]["request"]
    result = d["records"][0]["response"]["results"][0]
    declaration = d["artifact"]["program"]["derived"][0]
    if change == "entity":
        request["dataset"]["inputs"][0]["entity"] = "Family"
    elif change == "entity_id":
        request["dataset"]["inputs"][0]["entity_id"] = "other"
    elif change == "bool_int":
        request["dataset"]["inputs"][0]["value"]["value"] = 1
    elif change == "foreign_fact":
        request["dataset"]["inputs"][0]["name"] = (
            "other#input." + request["dataset"]["inputs"][0]["name"].split("#input.")[1]
        )
    elif change == "value":
        request["dataset"]["inputs"][1]["value"]["value"] = 801
    elif change == "frame":
        request["queries"][0]["period"]["end"] = "2026-12-31"
    elif change == "fallback":
        d["records"][0]["response"]["metadata"]["actual_mode"] = "evaluate"
    elif change == "extra_metadata":
        d["records"][0]["response"]["metadata"]["extra"] = True
    elif change == "response_value":
        next(iter(result["outputs"].values()))["value"] = {
            "kind": "integer",
            "value": 999,
        }
    elif change == "expression":
        next(iter(result["trace"].values()))["executed_expression"] = "0"
    elif change == "missing_trace":
        result["trace"].pop(next(iter(result["trace"])))
    elif change == "duplicate_fact":
        request["dataset"]["inputs"].append(request["dataset"]["inputs"][0])
    elif change == "compiled_branch":
        declaration["expr"]["then_expr"], declaration["expr"]["else_expr"] = (
            declaration["expr"]["else_expr"],
            declaration["expr"]["then_expr"],
        )
    elif change == "compiled_bool":
        declaration["expr"]["condition"]["right"]["value"]["value"] = 1
    elif change == "version_bool":
        declaration["versions"][0]["expr"]["condition"]["right"]["value"]["value"] = 1
    elif change == "cad_float":
        next(u for u in d["artifact"]["program"]["units"] if u["name"] == "CAD")[
            "minor_units"
        ] = 2.0
    elif change == "cad_duplicate":
        d["artifact"]["program"]["units"].append(
            next(u for u in d["artifact"]["program"]["units"] if u["name"] == "CAD")
        )
    elif change == "extra_rule":
        d["artifact"]["program"]["derived"].append(declaration)
    elif change == "missing_record":
        d["records"].pop()
    elif change == "changed_case":
        d["cases"][0]["name"] += "_different"
    elif change.startswith("missing_"):
        indices = {
            "missing_guide_pair": {17, 18},
            "missing_historical_provider": {6},
            "missing_equal_provider": {15, 16},
            "missing_upper_boundary": {7},
        }[change]
        # Keep case indices and raw actual queries: omission must refuse, never
        # relabel historical captures as newly executed cases.
        d["records"] = [r for r in d["records"] if r["case_index"] not in indices]
    assert observe(tmp_path, d, stale=change == "stale") is None


@pytest.mark.parametrize(
    "control,accepted",
    [
        ("negative_available", False),
        ("negative_current_paid", False),
        ("valid_partial", True),
    ],
)
def test_actual_partial_query_domain_controls(tmp_path, control, accepted):
    from axiom_encode.harness.same_case_execution import case_key

    d = data()
    controls = json.loads((FIXTURE.parent / "partial-query-controls.json").read_text())
    c = next(x for x in controls if x["name"] == control)
    d["cases"].append(c["case"])
    d["records"].append(
        {
            "case_index": 19,
            "case_sha256": case_key(c["case"]),
            "request": c["request"],
            "response": c["response"],
        }
    )
    assert (observe(tmp_path, d) is not None) is accepted


@pytest.mark.parametrize("change", ["module", "catalog", "catalog_extra"])
def test_compiled_context_identity(tmp_path, change):
    d = data()
    if change == "module":
        d["artifact"]["program"]["module"]["source_verification"]["source_sha256"] = (
            "0" * 64
        )
    elif change == "catalog":
        d["artifact"]["metadata"]["input_catalog"][0]["request_names"] = [
            "foreign#input.fake"
        ]
    else:
        d["artifact"]["metadata"]["input_catalog"].append(
            d["artifact"]["metadata"]["input_catalog"][0]
        )
    assert observe(tmp_path, d) is None


@pytest.mark.parametrize("removed", [{17, 18}, {6}, {15, 16}, {7}])
def test_required_evidence_cannot_be_omitted_from_case_set(tmp_path, removed):
    # Synthetic ID renumbering solely isolates the aggregate evidence guard.
    # Original fixture remains an exact historical capture, never overwritten.
    d = data()
    keep = [
        (c, r)
        for c, r in zip(d["cases"], d["records"])
        if r["case_index"] not in removed
    ]
    d["cases"], d["records"] = [], []
    for index, (case, record) in enumerate(keep, 1):
        record["case_index"] = index
        entity = f"case-{index}"
        for fact in record["request"]["dataset"]["inputs"]:
            fact["entity_id"] = entity
        record["request"]["queries"][0]["entity_id"] = entity
        result = record["response"]["results"][0]
        result["entity_id"] = entity
        for trace in result["trace"].values():
            trace["entity_id"] = entity
        d["cases"].append(case)
        d["records"].append(record)
    assert observe(tmp_path, d) is None


@pytest.mark.parametrize(
    "change",
    ["unknown_pin", "extra_local_unit", "altered_builtin", "duplicate_builtin"],
)
def test_compiler_unit_profile_is_closed_and_pin_bound(tmp_path, change):
    d = data()
    if change == "unknown_pin":
        d["engine_pin"] = "d" * 40
    elif change == "extra_local_unit":
        d["artifact"]["program"]["units"].append({"name": "unknown", "kind": "count"})
    elif change == "altered_builtin":
        d["artifact"]["program"]["units"][0]["minor_units"] = 0
    else:
        d["artifact"]["program"]["units"].append(d["artifact"]["program"]["units"][0])
    assert observe(tmp_path, d) is None


def analyzer(tmp_path, d, *, actual=True):
    from axiom_encode.harness.validator_pipeline import ValidatorPipeline

    pipeline = ValidatorPipeline(
        tmp_path / "rulespec-ca",
        tmp_path / "engine",
        local_corpus_release=None,
        enable_oracles=False,
        require_complete_source_unit=True,
    )
    return pipeline._complete_source_unit_issues(
        d["content"],
        validation_source_texts={CITATION: d["source"]},
        proof_source_texts={CITATION: d["source"]},
        test_cases=d["cases"],
        execution_evidence=evidence(tmp_path, d) if actual else None,
    )


def test_full_analyzer_exact_complete_observation(tmp_path):
    assert analyzer(tmp_path, data()) == []


def test_full_analyzer_without_observation_cannot_borrow_duty_pairs(tmp_path):
    issues = analyzer(tmp_path, data(), actual=False)
    text = "\n".join(map(str, issues))
    assert "Source-stated exceptions" in text
    assert "after 1997" in text
    assert "guide" in text


@pytest.mark.parametrize("change", ["actual_wrong_kind", "invalid_static_binding"])
def test_full_analyzer_known_source_invalid_evidence_keeps_owned_obligations(
    tmp_path, change
):
    import yaml

    d = data()
    if change == "actual_wrong_kind":
        d["records"][0]["request"]["dataset"]["inputs"][0]["entity"] = "Family"
    else:
        payload = yaml.safe_load(d["content"])
        payload["rules"][4]["versions"][0]["formula"] = (
            "is_completing_student_loan_interest_schedule"
        )
        d["content"] = yaml.safe_dump(payload, sort_keys=False)
    text = "\n".join(map(str, analyzer(tmp_path, d)))
    assert "Source-stated exceptions" in text
    assert "after 1997" in text and "guide" in text


def test_exact_owner_span_mapping_and_duplicate_allocation(tmp_path):
    from dataclasses import replace

    from axiom_encode.harness.source_completeness import (
        SourceStructureBranch,
        _unmatched_evidence_obligations,
    )
    from axiom_encode.harness.student_loan_schedule_evidence import (
        _student_loan_context,
    )

    d = data()
    context = _student_loan_context(
        d["source"], CITATION, d["content"], evidence(tmp_path, d), d["cases"]
    )
    assert context is not None and context.observation is not None
    a, b = context.span("current_completion")
    direct = SourceStructureBranch(
        (), "exception-clause", "source unit", d["source"][a:b], a, b
    )
    tail = d["source"].index("M -V")
    generic = replace(direct, start=tail, text=d["source"][tail:b])
    assert (
        context.branch_role(generic)
        == context.branch_role(direct)
        == "current_completion"
    )
    assert context.witnesses(generic) == context.witnesses(direct)
    assert (
        len(
            _unmatched_evidence_obligations(
                {generic: context.witnesses(generic), direct: context.witnesses(direct)}
            )
        )
        == 1
    )
    for start in [tail - 1, tail + 1, a - 1, a + 1]:
        wrong = replace(direct, start=start, text=d["source"][start:b])
        assert context.branch_role(wrong) is None and not context.witnesses(wrong)
    tokens = {}
    for role in [
        "current_completion",
        "prior_completion",
        "historical_reconstruction",
        "enclosure",
        "guide_referral",
    ]:
        x, y = context.span(role)
        branch = SourceStructureBranch(
            (), "exception-clause", "source unit", d["source"][x:y], x, y
        )
        tokens[role] = context.witnesses(branch)
    assert all(
        not left & right
        for i, left in enumerate(tokens.values())
        for right in list(tokens.values())[i + 1 :]
    )


def test_unknown_source_shape_does_not_acquire_context():
    from axiom_encode.harness.student_loan_schedule_evidence import (
        _student_loan_context,
    )

    d = data()
    assert (
        _student_loan_context(
            "Do not:\n\n" + d["source"], CITATION, d["content"], None, d["cases"]
        )
        is None
    )


@pytest.mark.parametrize("remove", [11, 12])
def test_missing_decision_or_calculation_control_reaches_pair_requirement(
    tmp_path, remove
):
    d = data()
    keep = [
        (c, r) for c, r in zip(d["cases"], d["records"]) if r["case_index"] != remove
    ]
    d["cases"], d["records"] = [], []
    for i, (case, record) in enumerate(keep, 1):
        entity = f"case-{i}"
        record["case_index"] = i
        for fact in record["request"]["dataset"]["inputs"]:
            fact["entity_id"] = entity
        record["request"]["queries"][0]["entity_id"] = entity
        record["response"]["results"][0]["entity_id"] = entity
        for trace in record["response"]["results"][0]["trace"].values():
            trace["entity_id"] = entity
        d["cases"].append(case)
        d["records"].append(record)
    assert observe(tmp_path, d) is None


def test_known_owners_never_enter_generic_exception_candidate_path(
    tmp_path, monkeypatch
):
    from axiom_encode.harness import source_completeness as sc

    calls = []
    original = sc._exception_witnesses_for_branch

    def record(branch, **kwargs):
        calls.append((branch.start, branch.end))
        return original(branch, **kwargs)

    monkeypatch.setattr(sc, "_exception_witnesses_for_branch", record)
    analyzer(tmp_path, data(), actual=False)
    assert not calls


def test_existing_extractor_has_one_key_per_physical_obligation():
    from axiom_encode.harness.source_completeness import _source_exception_branches

    d = data()
    branches = _source_exception_branches(
        d["source"],
        raw_source_text=d["source"],
        branches=(),
        active_branches=(),
        deferred_paths=set(),
        corpus_citation_path=CITATION,
    )
    keys = [(b.path, b.start, b.end) for b in branches]
    assert len(keys) == len(set(keys)) == 4
    assert [b.start for b in branches] == [76, 226, 387, 1214]
