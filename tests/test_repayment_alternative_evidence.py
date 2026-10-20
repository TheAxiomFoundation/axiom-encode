"""Alternative correspondence is source-bound; it is not runtime evidence."""

import copy
import json
from dataclasses import replace

import pytest

from axiom_encode.harness import source_completeness as s
from axiom_encode.harness import validator_pipeline as vp
from axiom_encode.harness.repayment_alternative_evidence import (
    certify_repayment_alternatives,
)
from axiom_encode.harness.validator_pipeline import ValidatorPipeline
from axiom_encode.printed_source_rows import recognize_printed_rows
from tests import test_imported_export_evidence as observations
from tests.release_object_fixtures import bind_test_corpus_release

BASE = observations.BASE


@pytest.fixture
def repayment(tmp_path):
    world = observations.world.__wrapped__(tmp_path)
    consumer, provider, context, root, path = world
    source = context[BASE + "/step-3"]
    structure = recognize_printed_rows(source)
    start, end = structure.atomic_spans[0]
    group = source[start:end]
    for name, coordinate in (("second", 11300), ("third", 14600)):
        rule = copy.deepcopy(provider["rules"][0])
        rule["name"] = name
        rule["versions"][0]["formula"] = name + "_fact"
        provider["inputs"].append(
            {
                "name": name + "_fact",
                "dtype": "Bool",
                "entity": "Person",
                "period": "Year",
            }
        )
        row = next(
            x
            for x in context[BASE + "/step-2"].splitlines()
            if x.startswith("Line ") and f"line {coordinate}:" in x
        )
        bullet = next(
            x[2:]
            for x in source.splitlines()
            if x.startswith("- You entered") and "11300" in x
        )
        rule["metadata"]["proof"]["atoms"][-2:] = [
            observations.atom("versions[0].formula", "formula", BASE + "/step-2", row),
            observations.atom(
                "versions[0].formula", "predicate", BASE + "/step-3", bullet
            ),
        ]
        provider["rules"].append(rule)
        consumer["imports"].append(observations.URI.replace("#observation", "#" + name))
    consumer["units"] = [{"name": "CAD", "kind": "currency", "minor_units": 2}]
    consumer["inputs"] = [
        {
            "name": "completed",
            "dtype": "Money",
            "unit": "CAD",
            "entity": "Person",
            "period": "Year",
        }
    ]
    owner = consumer["rules"][0]
    owner["versions"][0]["formula"] = (
        "if (observation and income > threshold_a) or ((second or third) and income > threshold_b): completed else: 0"
    )
    owner["metadata"]["proof"]["atoms"].append(
        observations.atom("versions[0].formula", "formula", BASE + "/step-3", group)
    )
    income = copy.deepcopy(owner)
    income["name"] = "income"
    income["versions"][0]["formula"] = "base_income - deduction"
    income["metadata"]["proof"]["atoms"][-1]["source"]["excerpt"] = next(
        x for x in source.splitlines() if x.startswith("Line 54 and line 23400:")
    )
    consumer["rules"].append(income)
    for name, value in (("threshold_a", 82125), ("threshold_b", 93454)):
        rule = copy.deepcopy(owner)
        rule["name"], rule["kind"] = name, "parameter"
        del rule["entity"], rule["period"]
        rule["versions"][0]["formula"] = value
        consumer["rules"].append(rule)
    return world


def records(world):
    first = observations.records(world)[0]
    return tuple(
        replace(
            first,
            import_uri=uri,
            canonical_target=uri,
            export_name=uri.rsplit("#", 1)[1],
        )
        for uri in world[0]["imports"]
    )


def coverage(world, supplied=None):
    raw = world[2][BASE + "/step-3"]
    return certify_repayment_alternatives(
        observations.dump(world[0]),
        source_text=raw.strip(),
        raw_source_text=raw,
        citation=BASE + "/step-3",
        source_context=world[2],
        imported_exports=records(world) if supplied is None else supplied,
        parse_branch=s._parse_first_formula_branch_node,
        parse_expression=s._parse_formula_expression,
    )


def test_exact_alternative_tree_and_source_offsets(repayment):
    found = coverage(repayment)
    assert len(found.records) == 1
    r = found.records[0]
    assert r.owner == "repayment"
    assert repayment[2][r.citation][slice(*r.raw_span)] == r.text
    assert r.tree[0] == "or"
    assert "11900" in repr(r.tree) and "14600" in repr(r.tree)


@pytest.mark.parametrize(
    "before,after",
    [
        (" or ((second", " and ((second"),
        ("second or third", "second and third"),
        (" > threshold_a", " >= threshold_a"),
        ("income > threshold_a", "threshold_a > income"),
        ("income > threshold_a", "income < threshold_a"),
        ("income > threshold_a", "income > threshold_b"),
        ("observation and income", "second and income"),
        ("second or third", "second"),
        ("second or third", "second or third or third"),
        ("completed else: 0", "0 else: completed"),
        ("completed else: 0", "completed else: 1"),
        ("completed else: 0", "completed else: False"),
        ("completed else: 0", "completed + 1 else: 0"),
        ("observation and", "True and"),
        ("observation and", "not observation and"),
        ("income > threshold_a", "income > 82125"),
    ],
)
def test_ast_refusals(repayment, before, after):
    rule = repayment[0]["rules"][0]
    formula = rule["versions"][0]["formula"]
    assert before in formula
    rule["versions"][0]["formula"] = formula.replace(before, after)
    assert not coverage(repayment).records


@pytest.mark.parametrize(
    "change",
    [
        "threshold_value",
        "threshold_unit",
        "threshold_date",
        "threshold_default",
        "income_coordinate",
        "income_owner",
        "income_default",
        "completed_default",
        "completed_type",
        "completed_period",
        "missing_completed",
        "missing_owner_proof",
        "partial_owner_proof",
        "owner_date",
        "extra_version",
        "unsupported_version",
        "provider_predicate",
        "wrong_unit_declaration",
    ],
)
def test_binding_refusals(repayment, change):
    c, p, context, _, _ = repayment
    owner, income, threshold = c["rules"][:3]
    if change == "threshold_value":
        threshold["versions"][0]["formula"] = 82126
    elif change == "threshold_unit":
        threshold["unit"] = "USD"
    elif change == "threshold_date":
        threshold["versions"][0]["effective_to"] = "2032-12-31"
    elif change == "threshold_default":
        threshold["default"] = 82125
    elif change == "income_coordinate":
        income["metadata"]["proof"]["atoms"][-1]["source"]["excerpt"] = next(
            x
            for x in context[BASE + "/step-3"].splitlines()
            if x.startswith("Line 56 and line 23600:")
        )
    elif change == "income_owner":
        income["entity"] = "Family"
    elif change == "income_default":
        income["default"] = 0
    elif change == "completed_default":
        c["inputs"][0]["default"] = 0
    elif change == "completed_type":
        c["inputs"][0]["dtype"] = "Bool"
    elif change == "completed_period":
        c["inputs"][0]["period"] = "Month"
    elif change == "missing_completed":
        c["inputs"] = []
    elif change == "missing_owner_proof":
        owner["metadata"]["proof"]["atoms"].pop()
    elif change == "partial_owner_proof":
        owner["metadata"]["proof"]["atoms"][-1]["source"]["excerpt"] = (
            'If not, enter "0" on line 23500.'
        )
    elif change == "owner_date":
        owner["versions"][0]["effective_to"] = "2031-11-30"
    elif change == "extra_version":
        owner["versions"].append(copy.deepcopy(owner["versions"][0]))
    elif change == "unsupported_version":
        owner["versions"][0]["unknown"] = True
    elif change == "provider_predicate":
        p["rules"][0]["metadata"]["proof"]["atoms"].pop()
    elif change == "wrong_unit_declaration":
        c["units"][0]["minor_units"] = 0
    assert not coverage(repayment).records


def test_commutative_order_preserves_all_roles(repayment):
    repayment[0]["rules"][0]["versions"][0]["formula"] = (
        "if (income > threshold_b and (third or second)) or (income > threshold_a and observation): completed else: 0"
    )
    assert coverage(repayment).records


def release_for(world):
    root = world[3].parent.parent / "corpus"
    path = root / "data/corpus/provisions/ca/policy/fixture.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "id": f"source-{i}",
            "jurisdiction": "ca",
            "document_class": "policy",
            "version": "fixture",
            "source_path": "sources/ca/policy/fixture/source.txt",
            "source_as_of": "2031-01-01",
            "expression_date": "2031-01-01",
            "citation_path": citation,
            "body": body,
        }
        for i, (citation, body) in enumerate(world[2].items())
    ]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return bind_test_corpus_release(
        root, "fixture-release", [("ca", "policy", "fixture")]
    )


def pipeline_issues(world, *, stale_imports=False):
    consumer, provider, context, root, path = world
    path.write_text(observations.dump(provider))
    consumer_path = path.with_name("step-3.yaml")
    consumer_path.write_text(observations.dump(consumer))
    content = observations.dump(consumer)
    if stale_imports:
        content = content.replace("step-2#observation", "step-9#observation")
    release = release_for(world)
    pipeline = ValidatorPipeline(
        root,
        root.parent / "engine",
        local_corpus_release=release,
        enable_oracles=False,
        require_complete_source_unit=True,
        source_citation_path=BASE + "/step-3",
        source_text=context[BASE + "/step-3"],
    )
    with vp._authoritative_corpus_scope(release):
        validation = pipeline._source_texts_for_rulespec_content(content)
        proofs = pipeline._proof_source_texts_for_rulespec_content(
            content, source_texts=validation
        )
        return pipeline._complete_source_unit_issues(
            content,
            validation_source_texts=validation,
            proof_source_texts=proofs,
            test_cases=[],
            rules_file=consumer_path,
        )


def opacity(issues):
    return [x for x in issues if "source-explicit-conditions" in x]


def test_normal_pipeline_consumes_only_exact_opacity_evidence(repayment):
    assert not opacity(pipeline_issues(repayment))
    assert pipeline_issues(repayment)  # No companion/runtime or other source clearance.
    repayment[1]["rules"][0]["metadata"]["proof"]["atoms"].pop()
    assert opacity(pipeline_issues(repayment))


def test_normal_pipeline_refuses_changed_consumer_imports(repayment):
    assert opacity(pipeline_issues(repayment, stale_imports=True))


def test_normal_pipeline_preserves_wrong_comparator_finding(repayment):
    owner = repayment[0]["rules"][0]
    owner["versions"][0]["formula"] = owner["versions"][0]["formula"].replace(
        " > threshold_a", " >= threshold_a"
    )
    assert opacity(pipeline_issues(repayment))


def test_coverage_refuses_different_consumer_or_clause(repayment):
    found = coverage(repayment)
    record = found.records[0]
    args = dict(
        owner=record.owner,
        consumer_content=observations.dump(repayment[0]),
        version=record.version,
        formula=repayment[0]["rules"][0]["versions"][0]["formula"],
        start=record.start,
        end=record.end,
        clause_start=record.span[0],
        clause_end=record.span[1],
        clause_text=record.text,
        source_text=repayment[2][record.citation].strip(),
        citation=record.citation,
    )
    assert found.owns_condition(**args)
    assert not found.owns_condition(
        **{**args, "consumer_content": args["consumer_content"] + "\n"}
    )
    assert not found.owns_condition(**{**args, "clause_start": record.span[0] + 1})
    assert not found.owns_condition(**{**args, "citation": BASE + "/foreign"})


def test_unexercised_version_requires_full_correspondence(repayment):
    consumer, provider, context, _, _ = repayment
    context[BASE + "/primary-document"] += "T1 2032\n"
    for payload in (consumer, provider):
        for rule in payload["rules"]:
            later = copy.deepcopy(rule["versions"][0])
            later["effective_from"], later["effective_to"] = "2032-01-01", "2032-12-31"
            rule["versions"].append(later)
            original = copy.deepcopy(rule["metadata"]["proof"]["atoms"])
            for atom in original:
                atom["path"] = atom["path"].replace("versions[0]", "versions[1]")
                if atom["kind"] == "effective_period":
                    atom["source"]["excerpt"] = "T1 2032"
            rule["metadata"]["proof"]["atoms"].extend(original)
    assert len(coverage(repayment).records) == 2
    consumer["rules"][0]["versions"][1]["formula"] = consumer["rules"][0]["versions"][
        1
    ]["formula"].replace(" > ", " >= ", 1)
    assert not coverage(repayment).records


def test_explicit_structural_conflict_never_builds_coverage(repayment, monkeypatch):
    source = repayment[2][BASE + "/step-3"]
    start = source.index("You entered")
    end = source.index("\n", start)
    branch = s.SourceStructureBranch(
        ("conflict",), "subsection", "conflict", source[start:end], start, end
    )
    monkeypatch.setattr(s, "recognize_source_structure", lambda *a, **kw: (branch,))

    def forbidden(*args, **kwargs):
        raise AssertionError(
            "conflicting structural ownership must not construct evidence"
        )

    monkeypatch.setattr(s, "certify_repayment_alternatives", forbidden)
    assert pipeline_issues(repayment)


def test_raw_whitespace_retains_absolute_proof_coordinates(repayment):
    for citation in (BASE + "/step-2", BASE + "/step-3"):
        repayment[2][citation] = "\n \n" + repayment[2][citation] + "\n  "
    rule = repayment[0]["rules"][0]
    rule["versions"][0]["formula"] = " \n" + rule["versions"][0]["formula"] + "\n "
    found = coverage(repayment)
    assert len(found.records) == 1
    r = found.records[0]
    assert r.raw_span[0] == r.span[0] + 3
    assert repayment[2][r.citation][slice(*r.raw_span)] == r.text
    assert len(r.observations) == 3 and r.local_proofs
    assert not opacity(pipeline_issues(repayment))


@pytest.mark.parametrize(
    "before,after",
    [("unexpected ", ""), ("", " + unexpected"), ("", " else: unexpected")],
)
def test_extra_formula_content_refuses(repayment, before, after):
    version = repayment[0]["rules"][0]["versions"][0]
    version["formula"] = before + version["formula"] + after
    assert not coverage(repayment).records


def test_stale_consumer_threshold_cannot_reuse_coverage(repayment):
    found = coverage(repayment)
    r = found.records[0]
    formula = repayment[0]["rules"][0]["versions"][0]["formula"]
    repayment[0]["rules"][2]["versions"][0]["formula"] = 82126
    assert not found.owns_condition(
        owner=r.owner,
        consumer_content=observations.dump(repayment[0]),
        version=r.version,
        formula=formula,
        start=r.start,
        end=r.end,
        clause_start=r.span[0],
        clause_end=r.span[1],
        clause_text=r.text,
        source_text=repayment[2][r.citation].strip(),
        citation=r.citation,
    )


def test_separate_provider_context_is_authenticated_and_does_not_widen_consumer(
    repayment, monkeypatch
):
    release = release_for(repayment)
    pipeline = ValidatorPipeline(
        repayment[3],
        repayment[3].parent / "engine",
        local_corpus_release=release,
        enable_oracles=False,
    )
    content = observations.dump(repayment[0])
    bindings = records(repayment)
    assert pipeline._complete_source_import_proof_context(bindings, {}) is None
    with vp._authoritative_corpus_scope(release):
        ordinary = pipeline._proof_source_texts_for_rulespec_content(
            content, source_texts=None
        )
        numeric = pipeline._numeric_source_texts_for_rulespec_content(
            content, source_texts=None
        )
        before = copy.deepcopy(ordinary)
        assert BASE + "/step-2" not in ordinary and BASE + "/step-2" not in numeric
        calls = []
        original = pipeline._proof_source_texts_for_rulespec_content

        def tracked(text, **kwargs):
            calls.append(text)
            return original(text, **kwargs)

        monkeypatch.setattr(
            pipeline, "_proof_source_texts_for_rulespec_content", tracked
        )
        imported = pipeline._complete_source_import_proof_context(bindings, ordinary)
        assert len(calls) == 1  # Three exports, one exact provider snapshot.
        assert imported[BASE + "/step-2"] == repayment[2][BASE + "/step-2"]
        assert ordinary == before and BASE + "/step-2" not in numeric
        conflict = {**ordinary, BASE + "/step-3": "other source bytes"}
        assert (
            pipeline._complete_source_import_proof_context(bindings, conflict) is None
        )
        assert conflict[BASE + "/step-3"] == "other source bytes"
        assert (
            pipeline._complete_source_import_proof_context(
                (replace(bindings[0], content_sha256="0" * 64),), ordinary
            )
            is None
        )
    unbound = ValidatorPipeline(
        repayment[3],
        repayment[3].parent / "engine",
        local_corpus_release=None,
        enable_oracles=False,
    )
    with vp._authoritative_corpus_scope(release):
        assert unbound._complete_source_import_proof_context(bindings, before) is None


def test_missing_provider_source_cannot_supply_context(repayment):
    release = release_for(repayment)
    provider = repayment[1]
    provider["rules"][0]["metadata"]["proof"]["atoms"][-1]["source"][
        "corpus_citation_path"
    ] = BASE + "/missing"
    pipeline = ValidatorPipeline(
        repayment[3],
        repayment[3].parent / "engine",
        local_corpus_release=release,
        enable_oracles=False,
    )
    with vp._authoritative_corpus_scope(release):
        assert (
            pipeline._complete_source_import_proof_context(records(repayment), {})
            is None
        )


def test_provider_authentication_failure_propagates(repayment, monkeypatch):
    release = release_for(repayment)
    pipeline = ValidatorPipeline(
        repayment[3],
        repayment[3].parent / "engine",
        local_corpus_release=release,
        enable_oracles=False,
    )

    def failed(*a, **kw):
        raise vp.CorpusResolutionError("fixture authentication failure")

    monkeypatch.setattr(pipeline, "_proof_source_texts_for_rulespec_content", failed)
    with (
        vp._authoritative_corpus_scope(release),
        pytest.raises(vp.CorpusResolutionError, match="authentication failure"),
    ):
        pipeline._complete_source_import_proof_context(records(repayment), {})
