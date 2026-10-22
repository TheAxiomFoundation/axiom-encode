"""Physical threshold identity precedes own-case boundary evidence."""

import functools
from dataclasses import replace
from decimal import Decimal

import pytest

from axiom_encode.harness import actual_repayment_boundary_evidence as boundary
from axiom_encode.harness import source_completeness as s
from axiom_encode.harness.validator_pipeline import (
    extract_typed_numeric_inventory_occurrences_from_text,
)
from tests import test_imported_export_evidence as observations
from tests.test_repayment_alternative_evidence import records, repayment  # noqa: F401


def binding(world):
    content = observations.dump(world[0])
    citation = observations.BASE + "/step-3"
    source = world[2][citation].strip()
    branch = s.SourceStructureBranch((), "source-unit", "root", source, 0, len(source))
    obligations = s._source_boundary_obligations(
        (branch,),
        source_text=source,
        extract_numeric_occurrences=functools.partial(
            extract_typed_numeric_inventory_occurrences_from_text, profile="en-US"
        ),
    )
    return dict(
        content=content,
        source_text=source,
        citation=citation,
        context=world[2],
        records=records(world),
        principal={r["name"]: r for r in world[0]["rules"]},
        boundary_branches=(branch,),
    ), obligations


def test_exact_physical_thresholds(repayment):  # noqa: F811
    args, obligations = binding(repayment)
    found = [
        r
        for branch, occurrence in obligations
        for r in boundary.bind_occurrences(branch=branch, boundary=occurrence, **args)
    ]
    assert {r.threshold for r in found} == {Decimal(82125), Decimal(93454)}
    assert {r.parameter for r in found} == {"threshold_a", "threshold_b"}
    for r in found:
        assert args["source_text"][slice(*r.span)].replace(",", "") == str(r.threshold)
        assert r.income == "income" and r.worksheet == "completed"


@pytest.mark.parametrize(
    "change", ["offset", "value", "source", "principal", "context", "conflict"]
)
def test_source_identity_refusals(repayment, change):  # noqa: F811
    args, obligations = binding(repayment)
    branch, occurrence = next((b, o) for b, o in obligations if o.value == 82125)
    assert boundary.bind_occurrences(branch=branch, boundary=occurrence, **args)
    if change == "offset":
        occurrence = replace(occurrence, start=occurrence.start + 1)
    elif change == "value":
        occurrence = replace(occurrence, value=93454)
    elif change == "source":
        args["source_text"] += " "
    elif change == "principal":
        args["principal"] = {**args["principal"], "invented": {"name": "invented"}}
    elif change == "context":
        args["context"] = {}
    else:
        group = boundary.bind_occurrences(branch=branch, boundary=occurrence, **args)[
            0
        ].record
        args["source_branches"] = (
            s.SourceStructureBranch(
                ("conflict",),
                "paragraph",
                "conflict",
                args["source_text"][group.span[0] + 10 : group.span[0] + 20],
                group.span[0] + 10,
                group.span[0] + 20,
            ),
        )
    assert not boundary.bind_occurrences(branch=branch, boundary=occurrence, **args)


@pytest.fixture
def actual(repayment, tmp_path):  # noqa: F811
    import copy
    import hashlib
    import json

    from axiom_encode.harness import same_case_execution as e

    consumer, provider, context, _, _ = repayment
    for name in ("base_income", "deduction"):
        consumer["inputs"].append(
            dict(name=name, dtype="Money", unit="CAD", entity="Person", period="Year")
        )
    args, obligations = binding(repayment)
    root = observations.BASE.replace("/policy/", ":policies/") + "/step-3"
    other = observations.URI.rsplit("#", 1)[0]
    snapshot = tmp_path / "runtime-snapshot"
    files = []
    for module, payload in ((root, consumer), (other, provider)):
        relative = module.replace(":", "/", 1) + ".yaml"
        original = tmp_path / "runtime-original" / relative
        original.parent.mkdir(parents=True, exist_ok=True)
        (snapshot / relative).parent.mkdir(parents=True, exist_ok=True)
        raw = observations.dump(payload).encode()
        original.write_bytes(raw)
        (snapshot / relative).write_bytes(raw)
        files.append(e.ClosureFile(relative, original, raw))
    exports = tuple(
        replace(
            r,
            resolved_path=str(files[1].original),
            jurisdiction_root=str(tmp_path / "runtime-original/ca"),
            content=files[1].raw.decode(),
            content_sha256=hashlib.sha256(files[1].raw).hexdigest(),
        )
        for r in args["records"]
    )
    derived = []
    for name in ("repayment", "income", "observation", "second", "third"):
        judgment = name in {"observation", "second", "third"}
        ref = (other if judgment else root) + "#" + name
        derived.append(
            dict(
                id=ref,
                entity="Person",
                period="Year",
                dtype="judgment" if judgment else "decimal",
                semantics="judgment" if judgment else "scalar",
                unit=None if judgment else "CAD",
            )
        )
    cases, captures = [], []
    for i, (income, payment) in enumerate(((82125, 0), (82125.01, 100)), 1):
        case = dict(
            name=f"case-{i}",
            entity_id="alice",
            period="2031-01-01",
            input={
                root + "#input.base_income": income,
                root + "#input.deduction": 0,
                other + "#input.observed": True,
                other + "#input.second_fact": False,
                other + "#input.third_fact": False,
            },
            output={root + "#repayment": payment, root + "#income": income},
        )
        if payment:
            case["input"][root + "#input.completed"] = payment
        period = dict(period_kind="tax_year", start="2031-01-01", end="2031-12-31")
        case["period"] = period.copy()
        query = dict(entity_id="alice", period=period, outputs=list(case["output"]))
        trace = {}
        for ref, amount in case["output"].items():
            trace[ref] = dict(
                id=ref,
                entity_id="alice",
                kind="scalar",
                dtype="decimal",
                unit="CAD",
                value=dict(kind="decimal", value=str(amount)),
            )
        trace[other + "#observation"] = dict(
            id=other + "#observation",
            entity_id="alice",
            kind="judgment",
            outcome="holds",
        )
        # The alternate disjunct was short-circuited: explicit direct facts,
        # not fabricated trace nodes, establish its unchanged false inputs.
        response = dict(
            metadata=dict(
                requested_mode="explain", actual_mode="explain", fallback_reason=None
            ),
            results=[
                dict(
                    entity_id="alice",
                    period=period,
                    outputs={k: copy.deepcopy(trace[k]) for k in case["output"]},
                    trace=trace,
                )
            ],
        )
        request = dict(mode="explain", queries=[query], dataset={})
        captures.append(
            e.CaseExecution(
                e.case_key(case), json.dumps(request), json.dumps(response), i
            )
        )
        cases.append(case)
    evidence = e.SameCaseEvidence(
        tuple(files),
        files[0].relative,
        snapshot,
        "a" * 40,
        "b" * 64,
        "c" * 40,
        json.dumps(dict(program=dict(derived=derived))).encode(),
        tuple(captures),
    )
    branch, occurrence = next((b, o) for b, o in obligations if o.value == 82125)
    return dict(
        evidence=evidence,
        branch=branch,
        boundary=occurrence,
        content=args["content"],
        source_text=args["source_text"],
        corpus_citation_path=args["citation"],
        imported_exports=exports,
        source_context=context,
        principal_rules={
            r["name"]: r for r in consumer["rules"] if r["kind"] == "derived"
        },
        asserted_by_rule={"repayment": cases, "income": cases},
        formula_environment={"threshold_a": 82125, "threshold_b": 93454},
        boundary_branches=args["boundary_branches"],
    )


def test_actual_sparse_threshold_pair(actual):
    assert len(boundary.witnesses(**actual)) == 1


@pytest.mark.parametrize(
    "change",
    [
        "extra-change",
        "extra-new",
        "inactive-worksheet",
        "missing-worksheet",
        "missing-bool",
        "numeric-bool",
        "wrong-unit",
        "wrong-year",
        "wrong-person",
        "stale-case",
        "missing-index",
        "missing-income-trace",
        "missing-local-trace",
        "conflicting-direct-trace",
        "listed-output",
        "wrong-row-output",
        "zero-payment",
        "alternate-active",
        "missing-income-fact",
        "wrong-parameter",
        "only-equal",
        "only-above",
    ],
)
def test_actual_pair_refusals(actual, change):
    import json

    from axiom_encode.harness import same_case_execution as e

    assert boundary.witnesses(**actual)
    cases = actual["asserted_by_rule"]["repayment"]
    root = actual["evidence"].root_relative.removesuffix(".yaml").replace("/", ":", 1)
    other = observations.URI.rsplit("#", 1)[0]
    evidence = actual["evidence"]
    captures = list(evidence.records)
    if change == "extra-change":
        cases[1]["input"][root + "#input.deduction"] = 1
    elif change == "extra-new":
        cases[1]["input"][root + "#input.unused"] = 1
    elif change == "inactive-worksheet":
        cases[0]["input"][root + "#input.completed"] = 0
    elif change == "missing-worksheet":
        cases[1]["input"].pop(root + "#input.completed")
    elif change in ("missing-bool", "numeric-bool"):
        cases[0]["input"].pop(other + "#input.third_fact")
        if change == "numeric-bool":
            cases[0]["input"][other + "#input.third_fact"] = 0
    elif change == "wrong-year":
        cases[0]["period"]["start"] = "2030-01-01"
    elif change == "wrong-person":
        cases[0]["entity_id"] = "bob"
    elif change == "stale-case":
        cases[0]["name"] = "changed"
        assert not boundary.witnesses(**actual)
        return
    elif change == "missing-index":
        captures[0] = replace(captures[0], case_index=None)
    elif change in {
        "missing-income-trace",
        "missing-local-trace",
        "conflicting-direct-trace",
    }:
        response = json.loads(captures[0].response_json)
        trace = response["results"][0]["trace"]
        if change == "conflicting-direct-trace":
            trace[other + "#third"] = dict(
                id=other + "#third", entity_id="alice", kind="judgment", outcome="holds"
            )
        else:
            trace.pop(
                root + ("#income" if change == "missing-income-trace" else "#repayment")
            )
        captures[0] = replace(captures[0], response_json=json.dumps(response))
    elif change in {"listed-output", "wrong-row-output"}:
        output = cases[0]["output"]
        cases[0]["output"] = [output] if change == "listed-output" else {"bob": output}
    elif change == "zero-payment":
        cases[1]["input"][root + "#input.completed"] = 0
    elif change == "alternate-active":
        # Own-case values still below the alternate threshold; changing the
        # unrelated Bool prevents the one-common-Money-only pair contract.
        cases[1]["input"][other + "#input.second_fact"] = True
    elif change == "missing-income-fact":
        cases[0]["input"].pop(root + "#input.base_income")
    elif change == "wrong-parameter":
        actual["formula_environment"]["threshold_a"] = 82124
    elif change in {"only-equal", "only-above"}:
        chosen = cases[0 if change == "only-equal" else 1]
        actual["asserted_by_rule"] = {"repayment": [chosen], "income": [chosen]}
    elif change == "wrong-unit":
        artifact = json.loads(evidence.artifact_bytes)
        artifact["program"]["derived"][0]["unit"] = "USD"
        evidence = replace(evidence, artifact_bytes=json.dumps(artifact).encode())
    actual["evidence"] = replace(
        evidence,
        records=tuple(
            replace(r, case_sha256=e.case_key(c)) for r, c in zip(captures, cases)
        ),
    )
    assert not boundary.witnesses(**actual)


def test_legacy_caches_and_cases_are_not_mutated(actual):
    import copy

    before = copy.deepcopy(
        {k: actual[k] for k in ("asserted_by_rule", "formula_environment")}
    )
    assert boundary.witnesses(**actual)
    assert before == {k: actual[k] for k in before}


def _oas_world(actual):
    import json

    from axiom_encode.harness import same_case_execution as e

    evidence = actual["evidence"]
    root = evidence.root_relative.removesuffix(".yaml").replace("/", ":", 1)
    other = observations.URI.rsplit("#", 1)[0]
    cases = actual["asserted_by_rule"]["repayment"]
    captures = []
    for i, (case, capture) in enumerate(zip(cases, evidence.records)):
        amount = 93454 if i == 0 else 93454.01
        case["input"][root + "#input.base_income"] = amount
        case["input"][other + "#input.observed"] = False
        case["input"][other + "#input.second_fact"] = True
        case["output"][root + "#income"] = amount
        response = json.loads(capture.response_json)
        row = response["results"][0]
        for where in (row["outputs"], row["trace"]):
            where[root + "#income"]["value"]["value"] = str(amount)
        row["trace"][other + "#observation"]["outcome"] = "not_holds"
        captures.append(
            replace(
                capture,
                case_sha256=e.case_key(case),
                response_json=json.dumps(response),
            )
        )
    actual["evidence"] = replace(evidence, records=tuple(captures))
    recall = s.authoritative_numeric_recall_text(actual["source_text"])
    actual["boundary"] = next(
        o
        for o in extract_typed_numeric_inventory_occurrences_from_text(
            recall, profile="en-US"
        )
        if o.value == 93454
    )
    return actual


def test_second_threshold_uses_own_direct_false_alternative(actual):
    assert boundary.witnesses(**_oas_world(actual))


def test_other_disjunct_already_active_refuses(actual):
    import json

    from axiom_encode.harness import same_case_execution as e

    _oas_world(actual)
    assert boundary.witnesses(**actual)
    other = observations.URI.rsplit("#", 1)[0]
    cases = actual["asserted_by_rule"]["repayment"]
    captures = []
    for case, capture in zip(cases, actual["evidence"].records):
        case["input"][other + "#input.observed"] = True
        response = json.loads(capture.response_json)
        response["results"][0]["trace"][other + "#observation"]["outcome"] = "holds"
        captures.append(
            replace(
                capture,
                case_sha256=e.case_key(case),
                response_json=json.dumps(response),
            )
        )
    actual["evidence"] = replace(actual["evidence"], records=tuple(captures))
    assert not boundary.witnesses(**actual)


def test_actual_analyzer_boundary_caller(actual):
    from axiom_encode.harness.validator_pipeline import (
        extract_named_scalar_occurrences,
        numeric_value_is_grounded,
    )

    def run(evidence):
        return s.analyze_complete_source_unit(
            actual["content"],
            actual["source_context"][actual["corpus_citation_path"]],
            corpus_citation_path=actual["corpus_citation_path"],
            test_cases=actual["asserted_by_rule"]["repayment"],
            extract_numeric_occurrences=functools.partial(
                extract_typed_numeric_inventory_occurrences_from_text, profile="en-US"
            ),
            extract_named_scalars=extract_named_scalar_occurrences,
            numeric_value_is_grounded=numeric_value_is_grounded,
            imported_export_bindings=actual["imported_exports"],
            imported_proof_source_context=actual["source_context"],
            source_context=actual["source_context"],
            execution_evidence=evidence,
        )

    before = [i for i in run(None).issues if "source-stated boundary" in i]
    after = [i for i in run(actual["evidence"]).issues if "source-stated boundary" in i]
    assert len(before) == len(after) == 1
    assert "82125" in before[0] and "93454" in before[0]
    assert "82125" not in after[0] and "93454" in after[0]


@pytest.mark.parametrize(
    "change",
    [
        "gte",
        "reverse",
        "wrong-result",
        "bad-other-version",
        "default",
        "foreign-proof",
        "borrowed-frame",
    ],
)
def test_current_source_snapshot_refusals(actual, change):
    import hashlib

    import yaml

    assert boundary.witnesses(**actual)
    payload = yaml.safe_load(actual["content"])
    owner = payload["rules"][0]
    if change == "gte":
        owner["versions"][0]["formula"] = owner["versions"][0]["formula"].replace(
            " > ", " >= "
        )
    elif change == "reverse":
        owner["versions"][0]["formula"] = owner["versions"][0]["formula"].replace(
            "income > threshold_a", "threshold_a > income"
        )
    elif change == "wrong-result":
        owner["versions"][0]["formula"] = owner["versions"][0]["formula"].replace(
            "completed else: 0", "0 else: completed"
        )
    elif change == "bad-other-version":
        owner["versions"].append(
            dict(effective_from="2032-01-01", effective_to="2032-12-31", formula="0")
        )
    elif change == "default":
        payload["inputs"][0]["default"] = 0
    elif change == "foreign-proof":
        owner["metadata"]["proof"]["atoms"][-1]["source"]["corpus_citation_path"] = (
            "ca/policy/other/step-3"
        )
    else:
        # A foreign header must not turn an unchanged observation wrapper into
        # a source-family-independent fact certificate.
        for atom in owner["metadata"]["proof"]["atoms"]:
            if atom["kind"] == "effective_period":
                atom["source"]["corpus_citation_path"] = "ca/policy/foreign/header"
        actual["source_context"]["ca/policy/foreign/header"] = "T1 2031"
    content = observations.dump(payload)
    actual["content"] = content
    actual["principal_rules"] = {
        r["name"]: r for r in payload["rules"] if r["kind"] == "derived"
    }
    evidence = actual["evidence"]
    root = evidence.files[0]
    root.original.write_text(content)
    (evidence.snapshot_root / root.relative).write_text(content)
    actual["evidence"] = replace(
        evidence, files=(replace(root, raw=content.encode()), *evidence.files[1:])
    )
    assert actual["evidence"].bound(content, actual["imported_exports"])
    assert (
        hashlib.sha256(content.encode()).digest() != hashlib.sha256(root.raw).digest()
    )
    assert not boundary.witnesses(**actual)


def test_same_numeric_value_at_foreign_occurrence_not_certified(actual):
    from types import SimpleNamespace

    assert boundary.witnesses(**actual)
    # The source parameter declaration supplies the same number, but cannot
    # manufacture a second physical obligation at a source coordinate.
    actual["boundary"] = SimpleNamespace(value=82125, start=0, end=5)
    assert not boundary.witnesses(**actual)


def test_missing_trace_alias_cannot_override_formula_environment(actual):
    assert boundary.witnesses(**actual)
    actual["formula_environment"]["third"] = False
    assert not boundary.witnesses(**actual)


def test_raw_outer_whitespace_keeps_exact_physical_source_identity(actual):
    assert boundary.witnesses(**actual)
    citation = actual["corpus_citation_path"]
    actual["source_context"][citation] = (
        "\n  " + actual["source_context"][citation] + " \n"
    )
    assert boundary.witnesses(**actual)


@pytest.mark.parametrize("collision", ["alias", "coordinate", "primitive"])
def test_duplicate_bound_observation_refuses(actual, monkeypatch, collision):
    # The normal binder refuses ambiguous provider metadata. Consumption also
    # checks immutable record multiplicity instead of allowing dict overwrite.
    assert boundary.witnesses(**actual)
    original = boundary.bind_occurrences

    def collided(*args, **kwargs):
        found = original(*args, **kwargs)
        result = []
        for b in found:
            observations = list(b.record.observations)
            first, second = observations[:2]
            key = {
                "alias": "export_uri",
                "coordinate": "coordinate",
                "primitive": "input_uri",
            }[collision]
            observations[1] = replace(second, **{key: getattr(first, key)})
            result.append(
                replace(b, record=replace(b.record, observations=tuple(observations)))
            )
        return tuple(result)

    monkeypatch.setattr(boundary, "bind_occurrences", collided)
    assert not boundary.witnesses(**actual)


@pytest.mark.parametrize(
    "which,raw",
    [
        ("request", "{"),
        ("response", "{"),
        ("artifact", "{"),
        ("request", "null"),
        ("response", "[]"),
        ("artifact", "[]"),
        ("request", '{"queries":[null]}'),
        ("response", '{"results":[null]}'),
        ("artifact", '{"program":{"derived":[null]}}'),
    ],
)
def test_malformed_optional_receipt_refuses(actual, which, raw):
    assert boundary.witnesses(**actual)
    evidence = actual["evidence"]
    if which == "artifact":
        evidence = replace(evidence, artifact_bytes=raw.encode())
    else:
        captures = list(evidence.records)
        captures[0] = replace(captures[0], **{which + "_json": raw})
        evidence = replace(evidence, records=tuple(captures))
    actual["evidence"] = evidence
    assert not boundary.witnesses(**actual)


@pytest.mark.parametrize("output", [None, [], 1, "output"])
def test_malformed_current_output_refuses(actual, output):
    from axiom_encode.harness import same_case_execution as e

    assert boundary.witnesses(**actual)
    cases = actual["asserted_by_rule"]["repayment"]
    cases[0]["output"] = output
    evidence = actual["evidence"]
    actual["evidence"] = replace(
        evidence,
        records=(
            replace(evidence.records[0], case_sha256=e.case_key(cases[0])),
            evidence.records[1],
        ),
    )
    assert not boundary.witnesses(**actual)
