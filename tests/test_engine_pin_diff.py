import copy
import json
import shutil
import subprocess
from collections import Counter
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from hypothesis import given, settings
from hypothesis import strategies as st

from axiom_encode import engine_pin_diff as diff

RULE = dict(
    module="dk/statutes/test.yaml",
    rule="amount",
    effective_from="2025-05-12",
    effective_to="2025-12-31",
)
VALUE = {"value": 1}
REFUSAL = {"error": "derived `amount` has no formula version at 2025-05-11"}


@pytest.mark.parametrize(
    "old,new,start,label",
    [
        (VALUE, VALUE, "2025-05-12", "PASS"),
        (VALUE, REFUSAL, "2025-05-11", "EXPECTED_PRESTART_REFUSAL"),
        (
            VALUE,
            {"error": "derived `amount` has no formula version at 2026-01-01"},
            "2026-01-01",
            "EXPECTED_PRESTART_REFUSAL",
        ),
        (VALUE, {"value": 2}, "2025-05-12", "DIFF"),
        (VALUE, {"error": "missing input"}, "2025-05-12", "DIFF"),
        ({"error": "old failure"}, REFUSAL, "2025-05-11", "DIFF"),
        (
            {"error": "parameter `rate` has no value for key `0` at 2025-05-11"},
            REFUSAL,
            "2025-05-11",
            "EXPECTED_PRESTART_REFUSAL",
        ),
        (
            {"error": "parameter `rate` has no value at 2025-05-11"},
            REFUSAL,
            "2025-05-11",
            "EXPECTED_PRESTART_REFUSAL",
        ),
        (
            {"error": "derived `upstream` has no formula version at 2025-05-11"},
            REFUSAL,
            "2025-05-11",
            "EXPECTED_PRESTART_REFUSAL",
        ),
        (
            {"error": "parameter `rate` has no value for key `0` at 2025-05-10"},
            REFUSAL,
            "2025-05-11",
            "DIFF",
        ),
        (
            {"error": "parameter `rate` has no value at 2025-05-11; then more"},
            REFUSAL,
            "2025-05-11",
            "DIFF",
        ),
        (
            VALUE,
            {"error": "derived `other` has no formula version at 2025-05-11"},
            "2025-05-11",
            "DIFF",
        ),
        (VALUE, REFUSAL, "2025-05-12", "DIFF"),
    ],
)
def test_classifier(old, new, start, label):
    assert diff.classify(old, new, [RULE], start) == label


def test_parity_classes():
    outcomes = dict(
        old=dict(explain=VALUE, fast={"value": 2}),
        new=dict(explain={"value": 3}, fast={"error": "failed"}),
    )
    assert {
        row["classification"]
        for row in diff.comparisons(outcomes, [RULE], "2025-05-12")
    } == {"DIFF", "PARITY_BREAK", "OLD_PARITY_BREAK"}


@pytest.mark.parametrize(
    "old_fast,new_explain,new_fast,expected",
    [
        ({"value": 2}, VALUE, VALUE, {"PASS", "OLD_PARITY_BREAK"}),
        (
            {"value": 2},
            {"value": 2},
            {"value": 2},
            {"DIFF", "PASS", "OLD_PARITY_BREAK"},
        ),
        (VALUE, VALUE, {"value": 2}, {"PASS", "PARITY_BREAK"}),
        (VALUE, REFUSAL, REFUSAL, {"PASS", "EXPECTED_PRESTART_REFUSAL"}),
    ],
)
def test_old_parity_is_not_a_gate_and_explain_differences_remain_visible(
    old_fast, new_explain, new_fast, expected
):
    outcomes = dict(
        old=dict(explain=VALUE, fast=old_fast),
        new=dict(explain=new_explain, fast=new_fast),
    )
    assert {
        row["classification"]
        for row in diff.comparisons(outcomes, [RULE], "2025-05-11")
    } == expected


def request(start="2025-01-01", length=30, offset=0, input_length=30):
    beginning = date.fromisoformat(start)
    return dict(
        mode="explain",
        queries=[
            dict(
                entity_id="case",
                outputs=["amount"],
                period=dict(
                    period_kind="custom",
                    name="test",
                    start=start,
                    end=(beginning + timedelta(days=length)).isoformat(),
                ),
            )
        ],
        dataset=dict(
            inputs=[
                dict(
                    name="input",
                    interval=dict(
                        start=(beginning + timedelta(days=offset)).isoformat(),
                        end=(
                            beginning + timedelta(days=offset + input_length)
                        ).isoformat(),
                    ),
                )
            ],
            relations=[],
        ),
    )


@given(
    delta=st.integers(-5000, 5000),
    offset=st.integers(-500, 500),
    length=st.integers(0, 500),
    input_length=st.integers(0, 1000),
    extra_intervals=st.lists(
        st.tuples(st.integers(-500, 500), st.integers(0, 1000)), max_size=20
    ),
)
@settings(deadline=None)
def test_shifting_preserves_coverage(
    delta, offset, length, input_length, extra_intervals
):
    original = request(length=length, offset=offset, input_length=input_length)
    for index, (extra_offset, extra_length) in enumerate(extra_intervals):
        item = request(offset=extra_offset, input_length=extra_length)["dataset"][
            "inputs"
        ][0]
        item["name"] = f"input_{index}"
        original["dataset"]["inputs"].append(item)
    shifted = diff.shift_request(original, delta)

    def covers(payload):
        period = payload["queries"][0]["period"]
        return [
            item["interval"]["start"] <= period["start"]
            and item["interval"]["end"] >= period["end"]
            for item in payload["dataset"]["inputs"]
        ]

    assert covers(original) == covers(shifted)
    assert diff.shift_request(shifted, -delta) == original


@given(st.lists(st.tuples(st.integers(0, 3), st.integers(0, 3)), max_size=30))
@settings(deadline=None)
def test_classification_order_independence(pairs):
    def classify(values):
        return Counter(
            diff.classify({"value": old}, {"value": new}, [RULE], "2025-05-12")
            for old, new in values
        )

    assert classify(pairs) == classify(reversed(pairs))


@given(
    st.lists(
        st.tuples(
            st.booleans(),
            st.integers(0, 3),
            st.booleans(),
            st.booleans(),
            st.booleans(),
        ),
        max_size=30,
    )
)
@settings(deadline=None)
def test_yaml_inventory_exactly_dated_single_derived(specs):
    rules, expected = [], []
    for index, (derived, version_count, sentinel, has_end, alias) in enumerate(specs):
        start = "0001-01-01" if sentinel else "2025-05-12"
        version = {"from" if alias else "effective_from": start}
        if has_end:
            version["to" if alias else "effective_to"] = "2025-12-31"
        rules.append(
            dict(
                name=f"rule_{index}",
                kind="derived" if derived else "parameter",
                versions=[copy.deepcopy(version) for _ in range(version_count)],
            )
        )
        if derived and version_count == 1 and (not sentinel or has_end):
            expected.append(f"rule_{index}")
    payload = yaml.safe_load(yaml.safe_dump(dict(rules=rules)))
    assert [rule["rule"] for rule in diff.inventory(payload, "module.yaml")] == expected


def test_current_period_leap_month():
    original = request(start="2023-02-01", length=27)
    original["queries"][0]["period"]["period_kind"] = "month"
    shifted = diff.current_request(original, date(2024, 2, 15))
    assert shifted["queries"][0]["period"]["start"] == "2024-02-01"
    assert shifted["queries"][0]["period"]["end"] == "2024-02-29"


def test_current_fiscal_tax_year():
    original = request(start="2025-04-06", length=364)
    original["queries"][0]["period"]["period_kind"] = "tax_year"
    shifted = diff.current_request(original, date(2026, 3, 1))
    assert shifted["queries"][0]["period"]["start"] == "2025-04-06"
    assert shifted["queries"][0]["period"]["end"] == "2026-04-05"
    shifted = diff.current_request(original, date(2026, 10, 27))
    assert shifted["queries"][0]["period"]["start"] == "2026-04-06"
    assert shifted["queries"][0]["period"]["end"] == "2027-04-05"


def test_shared_input_relation_and_query_intervals_shift_once():
    original = request()
    interval = original["dataset"]["inputs"][0]["interval"]
    original["dataset"]["inputs"].append(dict(name="other", interval=interval))
    original["dataset"]["relations"].append(dict(name="relation", interval=interval))
    original["queries"].append(copy.copy(original["queries"][0]))
    shifted = diff.shift_request(original, 1)
    assert all(
        item["interval"]["start"] == "2025-01-02"
        for item in shifted["dataset"]["inputs"] + shifted["dataset"]["relations"]
    )
    assert all(query["period"]["start"] == "2025-01-02" for query in shifted["queries"])
    assert diff.shift_request(shifted, -1) == original


def test_ambiguous_rule_names_do_not_hide_differences():
    unrelated = dict(
        RULE, module="dk/statutes/unrelated.yaml", effective_from="2030-01-01"
    )
    assert diff.classify(VALUE, REFUSAL, [RULE, unrelated], "2025-05-11") == "DIFF"
    artifact = dict(
        program=dict(derived=[dict(id=diff.qualified_rule(RULE), name="amount")])
    )
    assert (
        diff.classify(
            VALUE,
            REFUSAL,
            diff.inventory_context(artifact, [RULE, unrelated]),
            "2025-05-11",
        )
        == "EXPECTED_PRESTART_REFUSAL"
    )


def test_seed_prefers_dependent_output_over_unrelated_case():
    artifact = dict(
        program=dict(
            derived=[
                dict(id="dk:statutes/test#amount", name="amount", expr={}),
                dict(
                    id="dk:statutes/test#total",
                    name="total",
                    expr=dict(kind="derived", name="amount"),
                ),
            ]
        )
    )
    unrelated = dict(name="unrelated", output={"other": 1})
    dependent = dict(name="dependent", output={"dk:statutes/test#total": 1})
    direct = dict(name="direct", output={"dk:statutes/test#amount": 1})
    assert (
        diff.select_seed([unrelated, dependent], "dk:statutes/test#amount", artifact)
        is dependent
    )
    assert (
        diff.select_seed([dependent, direct], "dk:statutes/test#amount", artifact)
        is direct
    )


@pytest.mark.parametrize("reference", ["members", "dk:statutes/test#relation.members"])
def test_seed_follows_derived_relation_predicates(reference):
    output = "dk:statutes/test#eligible"
    artifact = dict(
        program=dict(
            derived=[
                dict(id=output, name="eligible", expr={}),
                dict(
                    id="dk:statutes/test#count",
                    name="count",
                    expr=dict(kind="count_related", relation=reference),
                ),
            ],
            relations=[
                dict(
                    name="members",
                    derivation=dict(
                        source_relation="dk:statutes/test#relation.raw",
                        predicate=dict(kind="literal"),
                    ),
                ),
                dict(
                    name="raw",
                    derivation=dict(
                        source_relation="base",
                        predicate=dict(kind="derived", name="eligible"),
                    ),
                ),
            ],
        )
    )
    unrelated = dict(name="unrelated", output={"other": 1})
    dependent = dict(name="dependent", output={"dk:statutes/test#count": 1})
    assert diff.select_seed([unrelated, dependent], output, artifact) is dependent


def test_parameter_errors_preserve_engine_parity_evidence():
    parameter_error = dict(rate={"error": "missing parameter index input"})
    outcomes = dict(
        old=dict(
            explain=dict(VALUE, companion_parameters=parameter_error),
            fast=dict(VALUE, companion_parameters=parameter_error),
        ),
        new=dict(
            explain=dict(VALUE, companion_parameters=parameter_error),
            fast=dict(value=2, companion_parameters=parameter_error),
        ),
    )
    rows = diff.comparisons(outcomes, [RULE], "2025-05-11")
    assert "PARITY_BREAK" in {row["classification"] for row in rows}
    assert outcomes["new"]["fast"]["value"] == 2


def test_parameter_change_cannot_be_hidden_by_expected_refusal():
    old = dict(VALUE, companion_parameters=dict(rate={"value": 1}))
    new = dict(REFUSAL, companion_parameters=dict(rate={"value": 2}))
    outcomes = dict(old=dict(explain=old, fast=old), new=dict(explain=new, fast=new))
    rows = diff.comparisons(outcomes, [RULE], "2025-05-11")
    assert {row["classification"] for row in rows} == {
        "PASS",
        "EXPECTED_PRESTART_REFUSAL",
        "DIFF",
    }
    assert diff.classify(old, new, [RULE], "2025-05-11") == "DIFF"
    assert not diff.companion_needs_repair("companion", rows, old)


def test_preexisting_parameter_errors_are_not_new_companion_failures():
    parameters = dict(rate={"error": "missing index"})
    old, new = (
        dict(VALUE, companion_parameters=parameters),
        dict(REFUSAL, companion_parameters=parameters),
    )
    outcomes = dict(old=dict(explain=old, fast=old), new=dict(explain=new, fast=new))
    rows = diff.comparisons(outcomes, [RULE], "2025-05-11")
    assert not diff.companion_needs_repair("companion", rows, old)


def test_refusal_with_another_error_is_not_expected():
    mixed = {"error": REFUSAL["error"] + "\nmissing input"}
    assert diff.classify(VALUE, mixed, [RULE], "2025-05-11") == "DIFF"


def test_report_includes_companion_refusal(tmp_path):
    outcomes = dict(
        old=dict(explain=VALUE, fast=VALUE), new=dict(explain=REFUSAL, fast=REFUSAL)
    )
    rows = diff.comparisons(outcomes, [RULE], "2025-05-11")
    report = dict(
        as_of="2026-10-27",
        engines=dict(old=dict(ref="a" * 40), new=dict(ref="b" * 40)),
        cases=[
            dict(
                module=RULE["module"],
                name="companion: day_before_commencement",
                output="amount",
                needs_encoder_repair=diff.companion_needs_repair(
                    "companion", rows, outcomes["old"]["explain"]
                ),
                outcomes=outcomes,
                comparisons=rows,
            )
        ],
    )
    counts = diff.write_report(report, tmp_path)
    assert counts == {"PASS": 2, "EXPECTED_PRESTART_REFUSAL": 1}
    saved = json.loads((tmp_path / "report.json").read_text())
    assert saved["cases"][0]["needs_encoder_repair"]
    assert "day_before_commencement" in (tmp_path / "report.md").read_text()


@pytest.mark.parametrize("label", diff.CLASSES)
def test_fake_executor_report_classes_and_whole_case(label, tmp_path):
    outcomes = dict(
        old=dict(explain=VALUE, fast=VALUE), new=dict(explain=VALUE, fast=VALUE)
    )
    if label == "EXPECTED_PRESTART_REFUSAL":
        outcomes["new"] = dict(explain=REFUSAL, fast=REFUSAL)
    elif label == "DIFF":
        outcomes["new"] = dict(explain={"value": 2}, fast={"value": 2})
    elif label == "PARITY_BREAK":
        outcomes["new"]["fast"] = {"error": "failed"}
    elif label == "OLD_PARITY_BREAK":
        outcomes["old"]["fast"] = {"error": "failed"}
    payload = request()
    payload["queries"][0]["outputs"] = ["amount", "total"]
    observed = []

    def executor(binary, artifact, engine_request, env):
        assert engine_request["queries"][0]["outputs"] == ["amount", "total"]
        observed.append((binary.name, engine_request["mode"]))
        return outcomes[binary.name][engine_request["mode"]]

    executed = {
        side: diff.run_modes(
            Path(side),
            Path("artifact"),
            payload,
            {},
            ("explain", "fast"),
            executor=executor,
        )
        for side in ("old", "new")
    }
    rows = diff.comparisons(executed, [RULE], "2025-05-11")
    report = dict(
        as_of="2026-10-27",
        engines=dict(old=dict(ref="a" * 40), new=dict(ref="b" * 40)),
        cases=[
            dict(
                module=RULE["module"],
                name="companion",
                output="amount,total",
                outcomes=executed,
                comparisons=rows,
            )
        ],
    )
    counts = diff.write_report(report, tmp_path)
    assert counts[label] > 0
    assert len(observed) == 4
    if label == "OLD_PARITY_BREAK":
        assert counts["DIFF"] == counts["PARITY_BREAK"] == 0


def test_execute_records_value_without_trace(monkeypatch):
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout=json.dumps(
                dict(
                    results=[
                        dict(
                            outputs=dict(
                                amount=dict(
                                    kind="scalar",
                                    dtype="Integer",
                                    value=dict(kind="integer", value=3),
                                    name="amount",
                                    id="amount",
                                )
                            ),
                            trace=dict(ignored=True),
                        )
                    ]
                )
            ),
            stderr="",
        ),
    )
    outcome = diff.execute(Path("engine"), Path("artifact"), request(), {})
    assert outcome["value"][0]["amount"]["value"]["value"] == 3
    assert "trace" not in json.dumps(outcome)


def test_bind_rejects_head_even_with_receipt(monkeypatch):
    monkeypatch.setattr(diff, "_git", lambda *args: "b" * 40)
    monkeypatch.setattr(
        diff,
        "resolve_pinned_engine_binary",
        lambda *args, **kwargs: pytest.fail(
            "must check HEAD before receipt resolution"
        ),
    )
    with pytest.raises(ValueError, match="HEAD"):
        diff.bind(Path("engine"), "a" * 40)


def test_shared_translator_table_parameter_indexes_and_errors(monkeypatch, tmp_path):
    from axiom_encode import cli

    monkeypatch.setattr(cli, "declared_input_entities", lambda *args, **kwargs: {})
    output = "dk:statutes/test#rate"
    artifact = dict(
        program=dict(
            derived=[],
            relations=[],
            parameters=[
                dict(
                    id=output,
                    name="rate",
                    indexed_by="index",
                    versions=[
                        dict(
                            effective_from="2025-01-01",
                            values={"2": dict(kind="integer", value=7)},
                        )
                    ],
                )
            ],
        )
    )
    case = dict(
        name="table index",
        period="2026-01",
        tables=dict(Person=[dict(index=2)]),
        output={output: 7},
    )
    payload = diff.build_request(
        tmp_path / "test.test.yaml", case, artifact, tmp_path, (tmp_path,)
    )
    assert payload["_companion_parameters"][output] == {
        "value": dict(kind="integer", value=7)
    }
    assert payload["queries"][0]["outputs"] == []
    case["tables"] = {}
    payload = diff.build_request(
        tmp_path / "test.test.yaml", case, artifact, tmp_path, (tmp_path,)
    )
    assert "error" in payload["_companion_parameters"][output]
    assert payload["queries"]


def test_end_to_end_same_engine(real_engine, tmp_path):
    checkout = tmp_path / "rulespec-de"
    root = checkout / "de"
    module = root / "statutes" / "estg" / "66.yaml"
    module.parent.mkdir(parents=True)
    fixtures = (
        Path(__file__).parent
        / "fixtures"
        / "source_completeness"
        / "estg_66_rounding_input_stage"
        / "minimally_fixed"
    )
    shutil.copyfile(fixtures / "66.yaml", module)
    shutil.copyfile(fixtures / "66.test.yaml", module.with_suffix(".test.yaml"))
    subprocess.run(["git", "init", str(checkout)], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(checkout), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(checkout),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-m",
            "Fixture",
        ],
        check=True,
        capture_output=True,
    )
    ref = diff._git(real_engine.root, "rev-parse", "HEAD")
    args = SimpleNamespace(
        root=root,
        paths=[],
        old_engine=real_engine.root,
        new_engine=real_engine.root,
        old_engine_ref=ref,
        new_engine_ref=ref,
        as_of=date(2026, 10, 27),
        out_dir=tmp_path / "report",
        rulespec_dependency_root=[],
    )
    assert diff.run(args) == 0
    report = json.loads((args.out_dir / "report.json").read_text())
    assert report["cases"]
    assert any(case["kind"] == "boundary" for case in report["cases"])
    assert any(
        "value" in case["outcomes"]["new"]["explain"] for case in report["cases"]
    )
    assert set(
        row["classification"] for case in report["cases"] for row in case["comparisons"]
    ) == {"PASS"}
