"""Source-owned sparse pairs require separately executed, exact branch worlds."""

import copy
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from axiom_encode.harness import actual_sparse_worksheet_evidence as sparse
from axiom_encode.harness import same_case_execution as e
from axiom_encode.harness import source_completeness as s
from axiom_encode.harness.imported_export_evidence import bind_entered_observations
from tests.test_actual_floor_evidence import pair as floor_pair  # noqa: F401
from tests.test_same_case_execution import ROOT, URI, world  # noqa: F401


@pytest.fixture
def pair(floor_pair):  # noqa: F811
    p = floor_pair
    fixture = Path(__file__).parent / "fixtures/form-numeric-locators"
    context = p["source_context"]
    citation = p["corpus_citation_path"]
    provider_citation = "ca/policy/test/provider"
    predicate_citation = "ca/policy/test/predicate"
    context[citation] = (fixture / "step-4.txt").read_text()
    context[provider_citation] = (fixture / "step-2.txt").read_text()
    context[predicate_citation] = (fixture / "step-3.txt").read_text()
    text = next(
        line
        for line in context[citation].splitlines()
        if line.startswith("Line 60 and")
    )
    payload = yaml.safe_load(p["content"])
    owner = payload["rules"][0]
    owner["versions"][0]["formula"] = "if entry: worksheet else: reported"
    owner["metadata"]["proof"]["atoms"][0]["source"] = {
        "corpus_citation_path": citation,
        "excerpt": text,
    }
    payload["inputs"].append(
        {
            "name": "worksheet",
            "dtype": "Money",
            "unit": "CAD",
            "entity": "Person",
            "period": "Year",
        }
    )
    selector_uri = URI.rsplit("#", 1)[0] + "#entry"
    payload["imports"].append(selector_uri)
    provider = yaml.safe_load(p["imported_exports"][0].content)
    reported = provider["rules"][0]
    reported["metadata"]["proof"]["atoms"][0]["source"]["excerpt"] = next(
        line
        for line in context[provider_citation].splitlines()
        if line.startswith("Line 32 and")
    )
    wrapper = copy.deepcopy(reported)
    wrapper.update(name="entry", dtype="Judgment")
    wrapper.pop("unit")
    wrapper["versions"][0]["formula"] = "entered"
    wrapper["metadata"]["proof"]["atoms"][0]["source"]["excerpt"] = next(
        line
        for line in context[provider_citation].splitlines()
        if line.startswith("Line 31 and")
    )
    quote = next(
        line[2:]
        for line in context[predicate_citation].splitlines()
        if line.startswith("- You entered an amount for OAS")
    )
    wrapper["metadata"]["proof"]["atoms"].append(
        {
            "path": "versions[0].formula",
            "kind": "predicate",
            "source": {"corpus_citation_path": predicate_citation, "excerpt": quote},
        }
    )
    provider["rules"].append(wrapper)
    provider["inputs"].append(
        {"name": "entered", "dtype": "Bool", "entity": "Person", "period": "Year"}
    )
    content = yaml.safe_dump(payload)
    provider_content = yaml.safe_dump(provider)
    ev = p["evidence"]
    root, other = ev.files
    for item, raw in [(root, content.encode()), (other, provider_content.encode())]:
        item.original.write_bytes(raw)
        (ev.snapshot_root / item.relative).write_bytes(raw)
    imports = tuple(payload["imports"])
    record = p["imported_exports"][0]
    record = replace(
        record,
        content=provider_content,
        content_sha256=hashlib.sha256(provider_content.encode()).hexdigest(),
        consumer_imports=imports,
    )
    records = (
        record,
        replace(
            record,
            import_uri=selector_uri,
            canonical_target=selector_uri,
            export_name="entry",
        ),
    )
    artifact = json.loads(ev.artifact_bytes)
    artifact["program"]["derived"].append(
        {
            "id": selector_uri,
            "entity": "Person",
            "period": "Year",
            "semantics": "judgment",
            "dtype": "judgment",
            "unit": None,
        }
    )
    cases = copy.deepcopy(p["asserted_by_rule"]["out"])
    captured = []
    for case, old, entered in zip(cases, ev.records, (False, True)):
        case["input"][URI.split("#")[0] + "#input.income"] = 3000
        case["input"][URI.split("#")[0] + "#input.entered"] = entered
        if entered:
            case["input"][ROOT + "#input.worksheet"] = 6000
        case["output"][ROOT + "#out"] = 6000 if entered else 3000
        response = json.loads(old.response_json)
        row = response["results"][0]
        for container in (row["outputs"], row["trace"]):
            container[ROOT + "#out"]["value"]["value"] = str(
                case["output"][ROOT + "#out"]
            )
        row["trace"][URI]["value"]["value"] = "3000"
        row["trace"][selector_uri] = {
            "id": selector_uri,
            "entity_id": row["entity_id"],
            "kind": "judgment",
            "outcome": "holds" if entered else "not_holds",
        }
        captured.append(
            replace(
                old, case_sha256=e.case_key(case), response_json=json.dumps(response)
            )
        )
    p.update(
        content=content,
        source_text=context[citation].strip(),
        branch=s.SourceStructureBranch(
            (),
            "formula-clause",
            "worksheet",
            text,
            context[citation].strip().index(text),
            context[citation].strip().index(text) + len(text),
        ),
        principal_rules={r["name"]: r for r in payload["rules"]},
        asserted_by_rule={"out": cases, "deduction": cases},
        imported_exports=records,
        evidence=replace(
            ev,
            files=(
                replace(root, raw=content.encode()),
                replace(other, raw=provider_content.encode()),
            ),
            artifact_bytes=json.dumps(artifact).encode(),
            records=tuple(captured),
        ),
    )
    p.pop("extract_numeric_occurrences")
    p.pop("numeric_value_is_grounded")
    return p


def test_exact_sparse_pair_and_unchanged_consumer_binder(pair):
    assert (
        bind_entered_observations(
            pair["content"],
            owner="out",
            consumer_citation=pair["corpus_citation_path"],
            source_context=pair["source_context"],
            imported_exports=pair["imported_exports"],
        )
        == ()
    )
    assert len(sparse.witnesses(**pair)) == 1


def _case_rebind(pair):
    cases = pair["asserted_by_rule"]["out"]
    ev = pair["evidence"]
    pair["evidence"] = replace(
        ev,
        records=tuple(
            replace(r, case_sha256=e.case_key(case))
            for case, r in zip(cases, ev.records)
        ),
    )


@pytest.mark.parametrize(
    "change",
    [
        "additional-change",
        "additional-new",
        "inactive-worksheet",
        "missing-active",
        "missing-bool",
        "truthy-bool",
        "same-bool",
        "stale-case",
        "no-index",
        "wrong-index",
        "wrong-entity",
        "wrong-year",
        "wrong-unit",
        "missing-selector-trace",
        "missing-amount-trace",
        "missing-local-trace",
        "listed-output",
        "wrong-row-output",
    ],
)
def test_pair_refusals(pair, change):
    assert sparse.witnesses(**pair)
    cases = pair["asserted_by_rule"]["out"]
    ev = pair["evidence"]
    records = list(ev.records)
    if change == "additional-change":
        cases[1]["input"][ROOT + "#input.local_amount"] = 6
    elif change == "additional-new":
        cases[1]["input"]["extra"] = 1
    elif change == "inactive-worksheet":
        cases[0]["input"][ROOT + "#input.worksheet"] = 6000
    elif change == "missing-active":
        cases[1]["input"].pop(ROOT + "#input.worksheet")
    elif change == "missing-bool":
        cases[0]["input"].pop(URI.split("#")[0] + "#input.entered")
    elif change == "truthy-bool":
        cases[0]["input"][URI.split("#")[0] + "#input.entered"] = 0
    elif change == "same-bool":
        cases[0]["input"][URI.split("#")[0] + "#input.entered"] = True
    elif change == "stale-case":
        cases[0]["name"] = "changed"
        assert not sparse.witnesses(**pair)
        return
    elif change in ("no-index", "wrong-index"):
        records[0] = replace(
            records[0], case_index=None if change == "no-index" else 99
        )
    elif change == "wrong-unit":
        artifact = json.loads(ev.artifact_bytes)
        artifact["program"]["derived"][0]["unit"] = "USD"
        pair["evidence"] = replace(ev, artifact_bytes=json.dumps(artifact).encode())
        assert not sparse.witnesses(**pair)
        return
    elif change in ("listed-output", "wrong-row-output"):
        cases[0]["output"][ROOT + "#out"] = (
            [{"person_id": "case-1", "value": 3000}]
            if change == "listed-output"
            else {"foreign": 3000}
        )
    else:
        response = json.loads(records[0].response_json)
        request = json.loads(records[0].request_json)
        if change == "wrong-entity":
            request["queries"][0]["entity_id"] = "foreign"
        elif change == "wrong-year":
            request["queries"][0]["period"]["start"] = "2030-01-01"
            response["results"][0]["period"] = request["queries"][0]["period"]
        else:
            key = (
                URI.rsplit("#", 1)[0] + "#entry"
                if change == "missing-selector-trace"
                else URI
                if change == "missing-amount-trace"
                else ROOT + "#deduction"
            )
            response["results"][0]["trace"].pop(key)
        records[0] = replace(
            records[0],
            request_json=json.dumps(request),
            response_json=json.dumps(response),
        )
    pair["evidence"] = replace(pair["evidence"], records=tuple(records))
    _case_rebind(pair)
    assert pair["evidence"].bound(pair["content"], pair["imported_exports"])
    assert not sparse.witnesses(**pair)


def _content_rebind(pair, payload):
    content = yaml.safe_dump(payload)
    ev = pair["evidence"]
    root = ev.files[0]
    root.original.write_text(content)
    (ev.snapshot_root / root.relative).write_text(content)
    pair.update(
        content=content,
        principal_rules={r["name"]: r for r in payload["rules"]},
        evidence=replace(
            ev, files=(replace(root, raw=content.encode()), *ev.files[1:])
        ),
    )


@pytest.mark.parametrize(
    "change",
    [
        "reverse-arms",
        "compound",
        "negative-selector",
        "prefix",
        "extra-version",
        "worksheet-default",
        "worksheet-computed",
        "other-worksheet-use",
        "wrong-proof",
        "wrong-frame",
        "wrong-currency",
        "wrong-owner",
    ],
)
def test_source_contract_refusals(pair, change):
    assert sparse.witnesses(**pair)
    payload = yaml.safe_load(pair["content"])
    owner = payload["rules"][0]
    formulas = {
        "reverse-arms": "if entry: reported else: worksheet",
        "compound": "if entry and True: worksheet else: reported",
        "negative-selector": "if not entry: worksheet else: reported",
        "prefix": "1 + (if entry: worksheet else: reported)",
    }
    if change in formulas:
        owner["versions"][0]["formula"] = formulas[change]
    elif change == "extra-version":
        owner["versions"].append(copy.deepcopy(owner["versions"][0]))
    elif change == "worksheet-default":
        payload["inputs"][-1]["default"] = 6000
    elif change == "worksheet-computed":
        payload["inputs"][-1]["formula"] = "6000"
    elif change == "other-worksheet-use":
        payload["rules"][1]["versions"][0]["formula"] = "worksheet"
    elif change == "wrong-proof":
        owner["metadata"]["proof"]["atoms"][0]["source"]["excerpt"] = (
            "Other payments deduction"
        )
    elif change == "wrong-frame":
        owner["versions"][0]["effective_to"] = "2032-12-31"
    elif change == "wrong-currency":
        owner["unit"] = "USD"
    elif change == "wrong-owner":
        owner["entity"] = "Family"
    _content_rebind(pair, payload)
    assert pair["evidence"].bound(pair["content"], pair["imported_exports"])
    assert not sparse.witnesses(**pair)


def _provider_rebind(pair, payload):
    content = yaml.safe_dump(payload)
    ev = pair["evidence"]
    old = ev.files[1]
    old.original.write_text(content)
    (ev.snapshot_root / old.relative).write_text(content)
    pair["evidence"] = replace(
        ev, files=(ev.files[0], replace(old, raw=content.encode()))
    )
    pair["imported_exports"] = tuple(
        replace(
            r,
            content=content,
            content_sha256=hashlib.sha256(content.encode()).hexdigest(),
        )
        for r in pair["imported_exports"]
    )
    assert pair["evidence"].bound(pair["content"], pair["imported_exports"])


@pytest.mark.parametrize(
    "change",
    [
        "missing-predicate",
        "wrong-predicate",
        "wrong-row",
        "default-bool",
        "computed-wrapper",
        "shared-bool-roles",
        "extra-version",
        "wrong-frame",
        "malformed-import",
        "direct-worksheet-import",
        "wrong-amount-row",
        "foreign-frame-family",
    ],
)
def test_provider_and_role_refusals(pair, change):
    assert sparse.witnesses(**pair)
    payload = yaml.safe_load(pair["imported_exports"][0].content)
    wrapper = payload["rules"][1]
    atoms = wrapper["metadata"]["proof"]["atoms"]
    if change == "missing-predicate":
        atoms.pop()
    elif change == "wrong-predicate":
        atoms[-1]["source"]["excerpt"] = "You entered an amount"
    elif change in ("wrong-row", "shared-bool-roles", "wrong-amount-row"):
        row = next(
            line
            for line in pair["source_context"]["ca/policy/test/provider"].splitlines()
            if line.startswith("Line 30 and")
        )
        if change == "wrong-row":
            atoms[0]["source"]["excerpt"] = row
        elif change == "wrong-amount-row":
            payload["rules"][0]["metadata"]["proof"]["atoms"][0]["source"][
                "excerpt"
            ] = row
        else:
            sibling = copy.deepcopy(wrapper)
            sibling["name"] = "another_entry"
            sibling["metadata"]["proof"]["atoms"][0]["source"]["excerpt"] = row
            payload["rules"].append(sibling)
    elif change == "default-bool":
        payload["inputs"][-1]["default"] = False
    elif change == "computed-wrapper":
        wrapper["versions"][0]["formula"] = "entered or True"
    elif change == "extra-version":
        wrapper["versions"].append(copy.deepcopy(wrapper["versions"][0]))
    elif change == "wrong-frame":
        wrapper["versions"][0]["effective_to"] = "2032-12-31"
    elif change == "malformed-import":
        payload["imports"] = ["ca:policies/test/../consumer#out"]
    elif change == "direct-worksheet-import":
        payload["imports"] = [ROOT + "#worksheet"]
    elif change == "foreign-frame-family":
        context = pair["source_context"]
        foreign = "ca/policy/foreign/header"
        context[foreign] = context["ca/policy/test/header"]
        consumer = yaml.safe_load(pair["content"])
        for module in (consumer, payload):
            for rule in module["rules"]:
                for atom in rule["metadata"]["proof"]["atoms"]:
                    if atom["kind"] == "effective_period":
                        atom["source"]["corpus_citation_path"] = foreign
        _content_rebind(pair, consumer)
    _provider_rebind(pair, payload)
    assert not sparse.witnesses(**pair)


def test_structural_conflict_and_internal_source_mutation(pair):
    assert sparse.witnesses(**pair)
    branch = pair["branch"]
    conflict = s.SourceStructureBranch(
        (), "explicit", "conflict", "amount", branch.start + 10, branch.start + 16
    )
    assert not sparse.witnesses(**pair, source_branches=(conflict,))
    key = pair["corpus_citation_path"]
    pair["source_context"][key] = pair["source_context"][key].replace(
        "Other payments deduction", "Another deduction"
    )
    assert not sparse.witnesses(**pair)


@pytest.mark.parametrize(
    "shape",
    ["null-rule-versions", "null-source", "null-module", "null-inputs", "null-imports"],
)
def test_optional_malformed_source_declines(pair, shape):
    payload = yaml.safe_load(pair["content"])
    if shape == "null-rule-versions":
        payload["rules"][1]["versions"] = None
    elif shape == "null-source":
        payload["rules"][0]["metadata"]["proof"]["atoms"][0]["source"] = None
    elif shape == "null-module":
        payload["module"] = None
    elif shape == "null-inputs":
        payload["inputs"] = None
    else:
        payload["imports"] = None
    _content_rebind(pair, payload)
    assert not sparse.witnesses(**pair)


def test_sparse_read_does_not_change_legacy_case_evaluation(pair):
    cases = pair["asserted_by_rule"]["out"]
    before = copy.deepcopy(cases)
    old = [
        s._case_dependency_environment(
            pair["principal_rules"],
            case,
            formula_environment={},
            require_asserted_value=False,
        )
        for case in cases
    ]
    assert sparse.witnesses(**pair)
    after = [
        s._case_dependency_environment(
            pair["principal_rules"],
            case,
            formula_environment={},
            require_asserted_value=False,
        )
        for case in cases
    ]
    assert after == old and cases == before
    assert all("out" not in value for value in after)


@pytest.mark.parametrize("values", [(1074, 82125, 93454), (1074.0, 82125.0, 93454.0)])
def test_closure_numeric_parameters_have_no_worksheet_references(pair, values):
    payload = yaml.safe_load(pair["imported_exports"][0].content)
    for index, value in enumerate(values):
        payload["rules"].append(
            {
                "name": f"threshold_{index}",
                "kind": "parameter",
                "dtype": "Money",
                "unit": "CAD",
                "versions": [{"formula": value}],
            }
        )
    _provider_rebind(pair, payload)
    assert sparse.witnesses(**pair)


@pytest.mark.parametrize("value", [True, None, [], {}, float("inf"), float("nan")])
def test_closure_unsupported_parameter_literals_decline(pair, value):
    payload = yaml.safe_load(pair["imported_exports"][0].content)
    payload["rules"].append(
        {"name": "threshold", "kind": "parameter", "versions": [{"formula": value}]}
    )
    _provider_rebind(pair, payload)
    assert not sparse.witnesses(**pair)


def _descriptive_role(pair, source, *, remove_proof=False, semantic=False):
    record = pair["imported_exports"][1]
    provider = yaml.safe_load(record.content)
    rule = next(r for r in provider["rules"] if r["name"] == record.export_name)
    frames = sparse._versions(rule, pair["source_context"])
    rule["source"] = source
    if remove_proof:
        rule["metadata"]["proof"]["atoms"] = [
            a for a in rule["metadata"]["proof"]["atoms"] if a["kind"] != "predicate"
        ]
    if semantic:
        rule["default"] = True
    content = yaml.safe_dump(provider)
    record = replace(
        record,
        content=content,
        content_sha256=hashlib.sha256(content.encode()).hexdigest(),
    )
    return sparse.bind_direct_entered_role(record, pair["source_context"], frames)


@pytest.mark.parametrize("source", [None, "", "  ", [], {}, True, 7])
def test_direct_role_malformed_descriptive_source_refuses(pair, source):
    assert _descriptive_role(pair, source) is None


def test_direct_role_description_is_not_proof(pair):
    assert _descriptive_role(pair, "Descriptive source only.") is not None
    assert (
        _descriptive_role(pair, "Descriptive source only.", remove_proof=True) is None
    )
    assert _descriptive_role(pair, "Descriptive source only.", semantic=True) is None


def test_exact_native_supplements_wrapper_description_binds(pair):
    fixtures = Path(__file__).parent / "fixtures/form-numeric-locators"
    provider = yaml.safe_load(
        (fixtures / "native-entered-observations.yaml").read_text()
    )
    base = "ca/policy/cra/t1-2025/t1-income-net-taxable-income"
    context = {
        base + "/step-2": (fixtures / "step-2.txt").read_text(),
        base + "/step-3": (fixtures / "step-3.txt").read_text(),
        base + "/primary-document": "T1 2025",
    }
    rule = next(r for r in provider["rules"] if "14600" in r["name"])
    content = yaml.safe_dump(provider)
    uri = base.replace("/policy/", ":policies/") + "/step-2#" + rule["name"]
    root = Path(pair["imported_exports"][1].jurisdiction_root)
    record = replace(
        pair["imported_exports"][1],
        import_uri=uri,
        canonical_target=uri,
        export_name=rule["name"],
        resolved_path=str(
            root / "policies/cra/t1-2025/t1-income-net-taxable-income/step-2.yaml"
        ),
        content=content,
        content_sha256=hashlib.sha256(content.encode()).hexdigest(),
        consumer_imports=(uri,),
    )
    role = sparse.bind_direct_entered_role(
        record, context, sparse._versions(rule, context)
    )
    assert role is not None and role.coordinate == 14600
