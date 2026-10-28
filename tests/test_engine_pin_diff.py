import copy
import hashlib
import json
import os
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


def test_identical_inventory_entries_are_one_rule():
    # A sibling dependency root re-lists its own checkout; a rule listed twice
    # is still one rule, while a conflicting date for that rule stays ambiguous.
    assert (
        diff.classify(VALUE, REFUSAL, [RULE, dict(RULE)], "2025-05-11")
        == "EXPECTED_PRESTART_REFUSAL"
    )
    conflicting = dict(RULE, effective_from="2025-05-11")
    for rules in ([RULE, conflicting], [conflicting, RULE]):
        assert diff.classify(VALUE, REFUSAL, rules, "2025-05-11") == "DIFF"


DATED_RULES = [
    RULE,
    dict(RULE, module="dk/statutes/unrelated.yaml", effective_from="2030-01-01"),
    dict(RULE, effective_from="2025-05-11"),
    dict(RULE, rule="other"),
]


@settings(deadline=None)
@given(
    old=st.sampled_from(
        [VALUE, {"error": "parameter `rate` has no value at 2025-05-11"}, REFUSAL]
    ),
    new=st.sampled_from([VALUE, REFUSAL, {"value": 2}]),
    start=st.sampled_from(["2025-05-11", "2025-05-12"]),
    rules=st.lists(st.sampled_from(DATED_RULES), max_size=6),
    data=st.data(),
)
def test_classification_depends_only_on_distinct_inventory_entries(
    old, new, start, rules, data
):
    # Neither duplicates nor the order of entries can change a classification.
    distinct = list({json.dumps(rule, sort_keys=True): rule for rule in rules}.values())
    reordered = data.draw(st.permutations(rules))
    expected = diff.classify(old, new, distinct, start)
    assert diff.classify(old, new, rules, start) == expected
    assert diff.classify(old, new, reordered, start) == expected
    assert diff.classify(old, new, list(reversed(distinct)), start) == expected


def test_sibling_dependency_roots_inventory_each_module_once(tmp_path):
    checkout, root, module = _calendar_regression_module(tmp_path)
    neighbour = checkout / "de-sn" / "statutes" / "estg" / "66.yaml"
    neighbour.parent.mkdir(parents=True)
    shutil.copyfile(module, neighbour)
    subprocess.run(["git", "init", "-q", str(checkout)], check=True)
    subprocess.run(["git", "-C", str(checkout), "add", "."], check=True)
    # run() resolves every dependency root to its checkout's top.
    modules, dated = diff.collect_modules(
        checkout, root, (checkout, checkout, checkout), []
    )
    assert [name for name, _, _ in modules] == ["de/statutes/estg/66.yaml"]
    entries = Counter((rule["module"], rule["rule"]) for rule in dated)
    assert set(entries.values()) == {1}
    assert {name for name, _ in entries} == {
        "de/statutes/estg/66.yaml",
        "de-sn/statutes/estg/66.yaml",
    }


def test_separate_checkouts_sharing_a_path_are_both_inventoried(tmp_path):
    # Dedup is by resolved file, never by relative name: another checkout's
    # file at the same path can carry other dates, and dropping it would let a
    # refusal inside its window pass as expected.
    checkout, root, module = _calendar_regression_module(tmp_path)
    other = tmp_path / "rulespec-de-fork"
    other_module = other / "de" / "statutes" / "estg" / "66.yaml"
    other_module.parent.mkdir(parents=True)
    payload = yaml.safe_load(module.read_text())
    for rule in payload["rules"]:
        for version in rule.get("versions", []):
            if version.get("effective_from") == "2025-01-01":
                version["effective_from"] = "2024-01-01"
    other_module.write_text(yaml.safe_dump(payload, sort_keys=False))
    for repo in (checkout, other):
        subprocess.run(["git", "init", "-q", str(repo)], check=True)
        subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    _, dated = diff.collect_modules(checkout, root, (checkout, other), [])
    windows = {}
    for rule in dated:
        assert rule["module"] == "de/statutes/estg/66.yaml"
        windows.setdefault(rule["rule"], set()).add(rule["effective_from"])
    moved = {name for name, starts in windows.items() if len(starts) == 2}
    assert moved
    name = sorted(moved)[0]
    assert windows[name] == {"2024-01-01", "2025-01-01"}
    refusal = {"error": f"derived `{name}` has no formula version at 2024-06-01"}
    candidates = [rule for rule in dated if rule["rule"] == name]
    assert diff.classify(VALUE, refusal, candidates, "2024-06-01") == "DIFF"


def test_each_file_is_parsed_once_however_many_roots_list_it(tmp_path, monkeypatch):
    checkout, root, module = _calendar_regression_module(tmp_path)
    neighbour = checkout / "de-sn" / "statutes" / "estg" / "66.yaml"
    neighbour.parent.mkdir(parents=True)
    shutil.copyfile(module, neighbour)
    subprocess.run(["git", "init", "-q", str(checkout)], check=True)
    subprocess.run(["git", "-C", str(checkout), "add", "."], check=True)
    parsed = Counter()
    load = diff.yaml.safe_load

    def counting_load(text):
        parsed[text] += 1
        return load(text)

    listed = []
    git = diff._git

    def counting_git(path, *args):
        if args[:1] == ("ls-files",):
            listed.append(Path(path).resolve())
        return git(path, *args)

    monkeypatch.setattr(diff.yaml, "safe_load", counting_load)
    monkeypatch.setattr(diff, "_git", counting_git)
    diff.collect_modules(checkout, root, (checkout,) * 5, [])
    # Both modules hold the same text; each file is read once, and the four
    # identical dependency roots are listed once.
    assert sum(parsed.values()) == 2
    assert listed == [checkout.resolve()] * 2


def test_reports_follow_the_held_output_directory(tmp_path):
    report = dict(
        as_of="2026-10-28",
        engines=dict(old=dict(ref="a" * 40), new=dict(ref="b" * 40)),
        cases=[],
        incomplete=[],
    )
    moved = tmp_path / "moved"
    with diff.ConfinedOutput(tmp_path / "out") as output:
        output.write(Path("artifacts/0.json"), b"{}")
        (tmp_path / "out").rename(moved)
        diff.write_report(report, output)
    assert not (tmp_path / "out").exists()
    assert sorted(p.name for p in moved.iterdir()) == [
        "artifacts",
        "report.json",
        "report.md",
    ]


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
    with diff.ConfinedOutput(tmp_path) as output:
        counts = diff.write_report(report, output)
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
    with diff.ConfinedOutput(tmp_path) as output:
        counts = diff.write_report(report, output)
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
    second = module.with_name("67.yaml")
    second.write_text(module.read_text())
    second.with_suffix(".test.yaml").write_text(
        module.with_suffix(".test.yaml").read_text().replace("estg/66#", "estg/67#")
    )
    subprocess.run(["git", "-C", str(checkout), "add", "."], check=True)
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
    from axiom_encode.harness import validator_pipeline

    scopes = {}

    def execute_with_scope(binary, artifact, payload, env):
        scope = validator_pipeline._RULESPEC_RESOLUTION_CACHE.get()
        assert scope is not None
        scopes.setdefault(artifact.name, []).append(scope)
        return diff.execute(binary, artifact, payload, env)

    assert diff.run(args, executor=execute_with_scope) == 0
    assert validator_pipeline._RULESPEC_RESOLUTION_CACHE.get() is None
    assert set(scopes) == {"0.json", "1.json"}
    assert all(all(scope is seen[0] for scope in seen) for seen in scopes.values())
    assert scopes["0.json"][0] is not scopes["1.json"][0]
    report = json.loads((args.out_dir / "report.json").read_text())
    assert len(report["modules"]) == 2
    assert report["cases"]
    assert any(case["kind"] == "boundary" for case in report["cases"])
    assert any(
        "value" in case["outcomes"]["new"]["explain"] for case in report["cases"]
    )
    assert set(
        row["classification"] for case in report["cases"] for row in case["comparisons"]
    ) == {"PASS"}


@given(
    kind=st.sampled_from(["month", "tax_year"]),
    seed_start=st.dates(min_value=date(1901, 1, 1), max_value=date(2096, 12, 31)),
    as_of=st.dates(min_value=date(1902, 1, 1), max_value=date(2097, 12, 31)),
    margins=st.lists(
        st.tuples(st.integers(0, 400), st.integers(0, 400)),
        min_size=1,
        max_size=10,
    ),
)
@settings(deadline=None)
def test_current_calendar_requests_preserve_input_and_relation_coverage(
    kind, seed_start, as_of, margins
):
    import calendar

    if kind == "month":
        seed_start = seed_start.replace(day=1)
        seed_end = seed_start.replace(
            day=calendar.monthrange(seed_start.year, seed_start.month)[1]
        )
    else:
        next_start = date(
            seed_start.year + 1,
            seed_start.month,
            min(
                seed_start.day,
                calendar.monthrange(seed_start.year + 1, seed_start.month)[1],
            ),
        )
        seed_end = next_start - timedelta(days=1)
    original = request(
        start=seed_start.isoformat(), length=(seed_end - seed_start).days
    )
    original["queries"][0]["period"]["period_kind"] = kind
    original["queries"].append(copy.copy(original["queries"][0]))
    for collection in ("inputs", "relations"):
        original["dataset"][collection] = [
            dict(
                name=f"{collection}_{index}",
                interval=dict(
                    start=(seed_start - timedelta(days=leading)).isoformat(),
                    end=(seed_end + timedelta(days=trailing)).isoformat(),
                ),
            )
            for index, (leading, trailing) in enumerate(margins)
        ]
    # The compact companion translator can alias input and relation intervals.
    original["dataset"]["relations"].append(original["dataset"]["inputs"][0])
    snapshot = copy.deepcopy(original)
    shifted = diff.current_request(original, as_of)
    assert original == snapshot
    query = shifted["queries"][0]["period"]
    assert query["start"] <= as_of.isoformat() <= query["end"]
    for collection in ("inputs", "relations"):
        for before, after in zip(
            original["dataset"][collection], shifted["dataset"][collection], strict=True
        ):
            interval = after["interval"]
            assert interval["start"] <= query["start"]
            assert interval["end"] >= query["end"]
            for endpoint, seed_endpoint in (("start", seed_start), ("end", seed_end)):
                assert (
                    date.fromisoformat(interval[endpoint])
                    - date.fromisoformat(query[endpoint])
                ) == (date.fromisoformat(before["interval"][endpoint]) - seed_endpoint)


def test_current_calendar_shared_query_and_fixture_intervals_translate_once():
    original = request(start="2026-02-01", length=27, input_length=27)
    interval = original["queries"][0]["period"]
    interval["period_kind"] = "month"
    original["dataset"]["inputs"][0]["interval"] = interval
    original["dataset"]["relations"].append(dict(name="members", interval=interval))
    original["queries"].append(copy.copy(original["queries"][0]))
    shifted = diff.current_request(original, date(2028, 2, 15))
    for item in shifted["dataset"]["inputs"] + shifted["dataset"]["relations"]:
        assert item["interval"]["start"] == "2028-02-01"
        assert item["interval"]["end"] == "2028-02-29"
    assert all(
        query["period"]["start"] == "2028-02-01"
        and query["period"]["end"] == "2028-02-29"
        for query in shifted["queries"]
    )


def _calendar_regression_module(tmp_path):
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
    return checkout, root, module


def test_current_leap_month_executes_with_real_engine(real_engine, tmp_path):
    import os

    from axiom_encode.rules_engine_compat import run_rulespec_compile

    checkout, root, module = _calendar_regression_module(tmp_path)
    artifact_path = tmp_path / "compiled.json"
    compiled = run_rulespec_compile(
        binary=real_engine.binary,
        program=module,
        rulespec_roots=[checkout],
        output=artifact_path,
        cwd=real_engine.root,
        env=os.environ.copy(),
    )
    assert compiled.returncode == 0, compiled.stderr or compiled.stdout
    artifact = json.loads(artifact_path.read_text())
    case = yaml.safe_load(module.with_suffix(".test.yaml").read_text())[0]
    case["period"] = "2026-02"
    payload = diff.build_request(
        module.with_suffix(".test.yaml"), case, artifact, root, (checkout,)
    )
    payload.pop("_companion_parameters", None)
    current = diff.current_request(payload, date(2028, 2, 15))
    assert current["queries"][0]["period"]["end"] == "2028-02-29"
    outcome = diff.execute(
        real_engine.binary, artifact_path, current, os.environ.copy()
    )
    assert "value" in outcome, outcome
    output = "de:statutes/estg/66#kindergeld_payable_for_month"
    assert outcome["value"][0][output]["outcome"] == "holds"


def test_identical_missing_input_refusals_are_incomplete(real_engine, tmp_path):
    checkout, root, _module = _calendar_regression_module(tmp_path)
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
        as_of=date(2028, 2, 15),
        out_dir=tmp_path / "report",
        rulespec_dependency_root=[],
    )
    failure = {"error": "missing input `required` over 2028-02-01..2028-02-29"}
    assert diff.run(args, executor=lambda *args: failure) == 1
    report = json.loads((args.out_dir / "report.json").read_text())
    assert report["incomplete"]
    assert report["counts"]["PASS"] == 0
    assert all("missing input" in item["error"] for item in report["incomplete"])
    assert report["counts"]["PASS"] == 0
    assert any(case["kind"] == "current" for case in report["cases"])


@pytest.mark.parametrize(
    "shape",
    [
        {"formula": "1", "effective_from": "2025-05-12", "effective_to": "2025-12-31"},
        {"formula": "1", "from": "2025-05-12", "to": "2025-12-31"},
        {"formula": "1", "from": "2025-05-12", "to": "2025-12-31", "versions": []},
        {
            "versions": [
                {
                    "formula": "1",
                    "effective_from": "2025-05-12",
                    "effective_to": "2025-12-31",
                }
            ]
        },
        {"versions": [{"formula": "1", "from": "2025-05-12", "to": "2025-12-31"}]},
        {
            "formula": "2",
            "effective_from": "2020-01-01",
            "versions": [{"formula": "1", "from": "2025-05-12", "to": "2025-12-31"}],
        },
    ],
)
def test_inventory_normalizes_engine_single_version_shapes(shape):
    rule = dict(name="amount", kind="derived", **shape)
    assert diff.inventory({"rules": [rule]}, RULE["module"]) == [RULE]


@pytest.mark.parametrize(
    "shape,expected_count",
    [
        ({"formula": "1"}, 1),
        ({"effective_from": "2025-05-12"}, 1),
        ({"from": "2025-05-12"}, 1),
        ({"effective_to": "2025-12-31"}, 0),
        ({"to": "2025-12-31"}, 0),
        ({"formula": None, "effective_from": None}, 0),
        ({"versions": []}, 0),
        ({"versions": [{}, {}], "formula": "1"}, 2),
    ],
)
def test_effective_version_normalization_matches_engine_conditions(
    shape, expected_count
):
    # Formula-only and date-only shorthand normalize here, just as in Rust;
    # compilation remains responsible for rejecting their missing required field.
    assert len(diff.effective_versions(shape)) == expected_count


JSON_SCALARS = st.none() | st.booleans() | st.integers() | st.text()
JSON_VALUES = st.recursive(
    JSON_SCALARS,
    lambda children: (
        st.lists(children, max_size=4)
        | st.dictionaries(st.text(), children, max_size=4)
    ),
    max_leaves=20,
)


@given(
    outcome=st.dictionaries(st.text(), JSON_VALUES, max_size=6),
    start=st.dates(min_value=date(1, 1, 1), max_value=date(9999, 12, 31)),
    rules=st.lists(st.just(RULE), max_size=4),
)
@settings(deadline=None)
def test_classifier_identity_for_generated_outcomes(outcome, start, rules):
    assert (
        diff.classify(outcome, copy.deepcopy(outcome), rules, start.isoformat())
        == "PASS"
    )


@pytest.mark.parametrize(
    "shorthand,sibling", [(False, False), (True, False), (False, True)]
)
def test_shorthand_inventory_and_boundaries_real_engine(
    real_engine, tmp_path, shorthand, sibling
):
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
    payload = yaml.safe_load((fixtures / "66.yaml").read_text())
    rule = next(
        rule
        for rule in payload["rules"]
        if rule["name"] == "kindergeld_payable_for_month"
    )
    if shorthand:
        rule.update(rule.pop("versions")[0])
        rule["effective_to"] = "2026-12-31"
    else:
        rule["versions"][0]["effective_to"] = "2026-12-31"
    payload["rules"] = [rule]
    module.write_text(yaml.safe_dump(payload, sort_keys=False))
    case = yaml.safe_load((fixtures / "66.test.yaml").read_text())[0]
    case["period"] = "2026-02"
    module.with_suffix(".test.yaml").write_text(yaml.safe_dump([case], sort_keys=False))
    dependencies = []
    if sibling:
        # A sibling subtree resolves to this same checkout, so it re-lists the
        # root's own module (the batch layout that hid Belgium's refusals).
        neighbour = checkout / "de-sn" / "statutes" / "estg" / "66.yaml"
        neighbour.parent.mkdir(parents=True)
        shutil.copyfile(fixtures / "66.yaml", neighbour)
        dependencies.append(neighbour.parents[2])
    subprocess.run(
        ["git", "init", "-q", str(checkout)], check=True, capture_output=True
    )
    subprocess.run(
        ["git", "-C", str(checkout), "add", "."], check=True, capture_output=True
    )
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
            "-qm",
            "Fixture",
        ],
        check=True,
        capture_output=True,
    )
    old_root = Path(
        os.environ.get("AXIOM_PIN_DIFF_OLD_ENGINE_ROOT", str(real_engine.root))
    )
    old_ref = diff._git(old_root, "rev-parse", "HEAD")
    new_ref = diff._git(real_engine.root, "rev-parse", "HEAD")
    historical = "AXIOM_PIN_DIFF_OLD_ENGINE_ROOT" in os.environ
    if historical:
        assert old_ref == "05eac9d2f89dabe5c6673176260762cef3a58f47"
        # The new side is the real_engine fixture, which require() holds to the pin.
        assert new_ref != old_ref
    args = SimpleNamespace(
        root=root,
        paths=[],
        old_engine=old_root,
        new_engine=real_engine.root,
        old_engine_ref=old_ref,
        new_engine_ref=new_ref,
        as_of=date(2026, 2, 15),
        out_dir=tmp_path / "shorthand-report",
        rulespec_dependency_root=dependencies,
    )
    assert diff.run(args) == 0
    report = json.loads((args.out_dir / "report.json").read_text())
    assert report["modules"][0]["compile_errors"] == {"old": None, "new": None}
    assert report["inventory"] == [
        dict(
            module="de/statutes/estg/66.yaml",
            rule="kindergeld_payable_for_month",
            effective_from="2025-01-01",
            effective_to="2026-12-31",
        )
    ]
    boundaries = {
        item["period_start"]: item
        for item in report["cases"]
        if item["kind"] == "boundary"
    }
    assert set(boundaries) == {
        "2024-12-31",
        "2025-01-01",
        "2025-01-02",
        "2026-12-30",
        "2026-12-31",
        "2027-01-01",
    }
    assert "value" in boundaries["2025-01-01"]["outcomes"]["new"]["explain"]
    if historical:
        before = boundaries["2024-12-31"]
        assert "value" in before["outcomes"]["old"]["explain"]
        assert (
            next(
                row["classification"]
                for row in before["comparisons"]
                if row["comparison"] == "old/new explain"
            )
            == "EXPECTED_PRESTART_REFUSAL"
        )


def _tracked_hashes(checkout):
    return {
        name: hashlib.sha256((checkout / name).read_bytes()).hexdigest()
        for name in diff._git(checkout, "ls-files", "-z").split("\0")
        if name
    }


def _confinement_checkout(base):
    checkout, root, module = _calendar_regression_module(base)
    subprocess.run(["git", "init", "-q", str(checkout)], check=True)
    subprocess.run(["git", "-C", str(checkout), "add", "."], check=True)
    return checkout, root, module


@pytest.mark.parametrize("target_checkout", ["root", "dependency"])
@pytest.mark.parametrize(
    "planted",
    [
        "report.json",
        "report.md",
        "artifacts/old",
        "artifacts/new/0.json",
        "artifacts/new/0.request.json",
    ],
)
def test_reused_output_links_preserve_every_tracked_byte(
    tmp_path, monkeypatch, planted, target_checkout
):
    checkout, root, module = _confinement_checkout(tmp_path / "main")
    dependency, dependency_root, dependency_module = _confinement_checkout(
        tmp_path / "dependency"
    )
    output = tmp_path / "evidence"
    link = output / planted
    link.parent.mkdir(parents=True)
    victim = module if target_checkout == "root" else dependency_module
    link.symlink_to(
        victim.parent
        if planted == "artifacts/old"
        else victim.with_suffix(".test.yaml")
    )
    before = {path: _tracked_hashes(path) for path in (checkout, dependency)}
    monkeypatch.setattr(
        diff,
        "bind",
        lambda *args: pytest.fail("Output admission must precede engine binding"),
    )
    args = SimpleNamespace(
        root=root,
        out_dir=output,
        rulespec_dependency_root=[dependency_root],
        paths=[],
        old_engine=checkout,
        new_engine=checkout,
        old_engine_ref="a" * 40,
        new_engine_ref="a" * 40,
    )
    try:
        with pytest.raises(ValueError, match="Symlink|Unsafe"):
            diff.run(args)
    finally:
        assert {path: _tracked_hashes(path) for path in before} == before


@pytest.mark.parametrize(
    "destination", ["root", "dependency", "linked-root", "linked-external"]
)
def test_out_dir_refuses_checkouts_and_symlinks(tmp_path, monkeypatch, destination):
    checkout, root, _ = _confinement_checkout(tmp_path / "main")
    dependency, dependency_root, _ = _confinement_checkout(tmp_path / "dependency")
    output = (
        root.parent / "evidence" if destination == "root" else dependency / "evidence"
    )
    if destination.startswith("linked"):
        target = checkout if destination == "linked-root" else tmp_path / "external"
        target.mkdir(exist_ok=True)
        output = tmp_path / "linked"
        output.symlink_to(target, target_is_directory=True)
    monkeypatch.setattr(
        diff,
        "bind",
        lambda *args: pytest.fail("Output admission must precede engine binding"),
    )
    args = SimpleNamespace(
        root=root,
        out_dir=output,
        rulespec_dependency_root=[dependency_root],
        paths=[],
        old_engine=checkout,
        new_engine=checkout,
        old_engine_ref="a" * 40,
        new_engine_ref="a" * 40,
    )
    with pytest.raises(ValueError, match="outside|Unsafe"):
        diff.run(args)


def test_output_atomic_replacement_preserves_hardlinked_source(tmp_path):
    checkout, _root, module = _confinement_checkout(tmp_path / "main")
    output = tmp_path / "evidence"
    output.mkdir()
    os.link(module, output / "report.json")
    before = _tracked_hashes(checkout)
    with diff.ConfinedOutput(output) as confined:
        confined.write(Path("report.json"), b"new report\n")
    assert _tracked_hashes(checkout) == before
    assert (output / "report.json").read_bytes() == b"new report\n"


def test_output_refuses_directory_link_planted_after_admission(tmp_path):
    checkout, _root, module = _confinement_checkout(tmp_path / "main")
    output = tmp_path / "evidence"
    before = _tracked_hashes(checkout)
    with diff.ConfinedOutput(output) as confined:
        (output / "artifacts").symlink_to(module.parent, target_is_directory=True)
        with pytest.raises(OSError):
            confined.write(Path("artifacts/66.yaml"), b"overwrite")
    assert _tracked_hashes(checkout) == before


def test_output_replaces_destination_link_without_following_race(tmp_path, monkeypatch):
    checkout, _root, module = _confinement_checkout(tmp_path / "main")
    output = tmp_path / "evidence"
    before = _tracked_hashes(checkout)
    original_replace = os.replace

    def plant_then_replace(source, destination, **kwargs):
        (output / destination).symlink_to(module)
        return original_replace(source, destination, **kwargs)

    monkeypatch.setattr(os, "replace", plant_then_replace)
    with diff.ConfinedOutput(output) as confined:
        confined.write(Path("report.json"), b"safe\n")
    assert _tracked_hashes(checkout) == before
    assert not (output / "report.json").is_symlink()


def test_run_retains_leap_day_fiscal_anchor(real_engine, tmp_path):
    checkout, root, module = _confinement_checkout(tmp_path)
    case = yaml.safe_load(module.with_suffix(".test.yaml").read_text())[0]
    case["period"] = {
        "period_kind": "tax_year",
        "start": "2024-02-29",
        "end": "2025-02-27",
    }
    module.with_suffix(".test.yaml").write_text(yaml.safe_dump([case]))
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
            "-am",
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
        as_of=date(2027, 3, 1),
        out_dir=tmp_path / "report",
        rulespec_dependency_root=[],
    )
    assert diff.run(args, executor=lambda *args: VALUE) == 0
    report = json.loads((args.out_dir / "report.json").read_text())
    current = next(case for case in report["cases"] if case["kind"] == "current")
    for payload in current["requests"].values():
        assert payload["queries"][0]["period"]["start"] == "2027-02-28"
        assert payload["queries"][0]["period"]["end"] == "2028-02-28"
        assert all(
            item["interval"]["end"] >= "2028-02-28"
            for item in payload["dataset"]["inputs"]
        )


def test_output_parent_components_refused_before_creating_checkout_children(tmp_path):
    checkout, root, _ = _confinement_checkout(tmp_path)
    output = checkout / "planted" / ".." / ".." / "evidence"
    assert not output.resolve().is_relative_to(checkout)
    args = SimpleNamespace(root=root, out_dir=output, rulespec_dependency_root=[])
    with pytest.raises(ValueError, match="parent component"):
        diff.run(args)
    assert not (checkout / "planted").exists()


def test_temporary_root_in_checkout_is_refused_without_writability_probe(
    tmp_path, monkeypatch
):
    checkout, root, _ = _confinement_checkout(tmp_path)
    monkeypatch.setenv("TMPDIR", str(checkout))
    monkeypatch.setattr(
        diff.tempfile,
        "gettempdir",
        lambda: pytest.fail("Do not probe writability inside a checkout"),
    )
    output = tmp_path / "evidence"
    args = SimpleNamespace(root=root, out_dir=output, rulespec_dependency_root=[])
    with pytest.raises(ValueError, match="temporary directory"):
        diff.run(args)
    assert not output.exists()


def test_checkout_confinement_checks_ancestor_identity(tmp_path, monkeypatch):
    checkout = tmp_path / "rulespec-de"
    checkout.mkdir()
    alias = tmp_path / "RULESPEC-DE"
    output = alias / "not-created" / "evidence"
    original_samefile = Path.samefile

    def samefile(path, other):
        if path == alias and other == checkout:
            return True
        return original_samefile(path, other)

    monkeypatch.setattr(Path, "samefile", samefile)
    assert diff.inside_checkout(output, (checkout,))
    assert not output.exists()


def test_request_translation_has_a_fresh_resolution_scope(monkeypatch, tmp_path):
    from axiom_encode import cli
    from axiom_encode.harness import validator_pipeline

    scopes = []

    def translate(*args, **kwargs):
        scope = validator_pipeline._RULESPEC_RESOLUTION_CACHE.get()
        assert scope is not None
        scopes.append(scope)
        return request()

    monkeypatch.setattr(cli, "_execute_rulespec_test_case", translate)
    artifact = {"program": {"derived": [], "parameters": [], "relations": []}}
    for _ in range(2):
        assert (
            diff.build_request(
                tmp_path / "test.test.yaml", {}, artifact, tmp_path, (tmp_path,)
            )
            == request()
        )
        assert validator_pipeline._RULESPEC_RESOLUTION_CACHE.get() is None
    assert scopes[0] is not scopes[1]
