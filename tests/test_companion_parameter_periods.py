"""A companion assertion never answers what the engine would refuse."""

from __future__ import annotations

import copy
import json
import os
import re
import subprocess
import tempfile
from collections.abc import Iterator
from datetime import date, timedelta
from itertools import product
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from axiom_encode import cli
from axiom_encode.companion_parameters import (
    rulespec_engine_date,
    rulespec_engine_index,
    rulespec_index_input,
    rulespec_parameter_version_at,
)
from axiom_encode.harness.validator_pipeline import ValidatorPipeline
from axiom_encode.rules_engine_compat import run_rulespec_compile

MODULE = "us:statutes/example"
# Consecutive days, so generated bounds and queried periods collide.
DAYS = [date(2020, 1, 1) + timedelta(days=offset) for offset in range(6)]
# Literal values lower to `program.parameters`, which the runners answer without
# the engine; computed ones lower to `program.derived`, which only the engine reads.
SHAPES = {
    "literal_integer": ("Integer", "6", 6, "parameters"),
    "literal_rate": ("Rate", "0.5", 0.5, "parameters"),
    "computed_integer": ("Integer", "3 + 3", 6, "derived"),
    "computed_rate": ("Rate", "1 / 2", 0.5, "derived"),
}
ENGINE_REFUSED_PERIODS = [
    pytest.param(
        {"period_kind": "tax_year", "start": "20200101", "end": "2020-12-31"},
        id="basic-date",
    ),
    pytest.param(
        {"period_kind": "tax_year", "start": "2020-W01-3", "end": "2020-12-31"},
        id="week-date",
    ),
    pytest.param(
        {"period_kind": "tax_year", "start": "2020-01-01", "end": "never"},
        id="unreadable-end",
    ),
    pytest.param(
        {"period_kind": "year", "start": "2020-01-01", "end": "2020-12-31"},
        id="unknown-kind",
    ),
    pytest.param(
        {"period_kind": "custom", "start": "2020-01-01", "end": "2020-12-31"},
        id="custom-without-name",
    ),
    pytest.param(
        {
            "period_kind": "custom",
            "name": 2020,
            "start": "2020-01-01",
            "end": "2020-12-31",
        },
        id="custom-with-a-number-for-a-name",
    ),
]
ABSENT = object()
# Crossed in full; the first date in each list and the text names are readable.
PERIOD_KINDS = ["month", "benefit_week", "tax_year", "custom", "year", "day"]
PERIOD_DATES = [
    "2020-01-01",
    date(2020, 12, 31),
    "20200101",
    "2020-W01-3",
    "2021-02-29",
    "never",
    None,
    20200101,
]
PERIOD_NAMES = [
    ABSENT,
    None,
    "day",
    "2020",
    "",
    2020,
    True,
    1.5,
    date(2020, 1, 1),
    ["day"],
    {"day": 1},
]
TABLE = {0: 10, 1: 7, 2: 9, 10**15: 12, 10**16: 11}
# Each index with the cell it reads, or None when the engine refuses it. The
# refused ones assert cell 1, which the truncating reader used to answer.
INDEXES = {
    "zero": (0, 10),
    "one": (1, 7),
    "two": (2, 9),
    "whole_decimal": (1.0, 7),
    "big_decimal": (1e15, 12),
    "exponent_decimal": (1e16, None),
    "fraction": (1.5, None),
    "high_fraction": (1.9, None),
    "missing_key": (99, None),
    "negative": (-1, None),
    "flag": (True, None),
    "text": ("1", None),
    "word": ("one", None),
}
# Each key a case might give a table's index under, with whether a direct read
# takes it. The engine takes only the first; rulespec-uk cases use the second.
INDEX_KEYS = {
    "engine_name": (f"{MODULE}#input.band", True),
    "dummy": ("dummy#input.band", True),
    "bare": ("band", False),
    "no_input_segment": (f"{MODULE}#band", False),
    "dotted": (f"{MODULE}.band", False),
    "input_prefix": ("input.band", False),
    "other_module": ("us:statutes/other#input.band", False),
    "longer_slot": (f"{MODULE}#input.income_band", False),
}


@st.composite
def version_lists(draw, min_size: int = 0, loadable: bool = False) -> list[dict]:
    """Version layouts; `loadable` keeps those the engine's loader accepts.

    The loader wants an explicit end before the next start, and lets a later
    version supersede an open one, so a bounded version can sit inside it.
    """
    starts = sorted(
        draw(
            st.lists(st.sampled_from(DAYS), min_size=min_size, max_size=4, unique=True)
        )
    )
    versions = []
    for value, start in enumerate(starts):
        later = starts[value + 1 :]
        limit = later[0] if loadable and later else date.max
        end = draw(
            st.none() | st.sampled_from([day for day in DAYS if start <= day < limit])
        )
        version = {
            "effective_from": start.isoformat(),
            "values": {"0": {"kind": "integer", "value": value}},
        }
        if end is not None:
            version["effective_to"] = end.isoformat()
        versions.append(version)
    return versions


def _covers(version: dict, day: date) -> bool:
    return (
        version["effective_from"]
        <= day.isoformat()
        <= version.get("effective_to", "9999-12-31")
    )


def _tax_year(year: int) -> dict[str, str]:
    return {
        "period_kind": "tax_year",
        "start": f"{year}-01-01",
        "end": f"{year}-12-31",
    }


def _day_period(day: date) -> dict[str, str]:
    return {
        "period_kind": "custom",
        "name": "day",
        "start": day.isoformat(),
        "end": day.isoformat(),
    }


def _pipeline(root: Path, engine_root: Path | None = None) -> ValidatorPipeline:
    return ValidatorPipeline(
        policy_repo_path=root,
        axiom_rules_path=engine_root or root,
        local_corpus_release=None,
        enable_oracles=False,
        enforce_repository_layout=False,
    )


def _outcome(answer) -> object:
    try:
        return answer()
    except ValueError as error:
        assert "has no version at" in str(error) or "has no value for key" in str(error)
        return "refused"


@pytest.fixture(scope="module")
def validator(tmp_path_factory) -> ValidatorPipeline:
    return _pipeline(tmp_path_factory.mktemp("rulespec-us") / "us")


@given(versions=version_lists(), day=st.sampled_from(DAYS))
def test_selection_reads_the_latest_covering_version_or_refuses(versions, day):
    covering = [version for version in versions if _covers(version, day)]
    parameter = {"name": "rate", "versions": versions}
    if not covering:
        with pytest.raises(ValueError, match="`rate` has no version at"):
            rulespec_parameter_version_at(parameter, day.isoformat())
        return
    chosen = rulespec_parameter_version_at(parameter, day.isoformat())
    assert _covers(chosen, day)
    assert chosen["effective_from"] == max(
        version["effective_from"] for version in covering
    )
    reordered = {"name": "rate", "versions": versions[::-1]}
    assert rulespec_parameter_version_at(reordered, day.isoformat()) == chosen


@given(versions=version_lists(), day=st.sampled_from(DAYS))
def test_both_runners_answer_alike(validator, versions, day):
    parameter = {"name": "rate", "versions": versions}
    period = _day_period(day)
    assert _outcome(
        lambda: cli._rulespec_parameter_value(parameter, {}, period["start"])
    ) == _outcome(
        lambda: validator._rulespec_compiled_parameter_value(parameter, period)
    )


def test_tied_versions_read_the_last_listed_as_the_engine_does():
    versions = [
        {"effective_from": "2020-01-01", "values": {"0": "first"}},
        {"effective_from": "2020-01-01", "values": {"0": "last"}},
    ]
    chosen = rulespec_parameter_version_at({"versions": versions}, "2020-06-01")
    assert chosen["values"] == {"0": "last"}


def test_an_ended_version_does_not_shadow_an_open_earlier_one():
    versions = [
        {"effective_from": "2000-01-01", "values": {"0": "open"}},
        {
            "effective_from": "2010-01-01",
            "effective_to": "2012-12-31",
            "values": {"0": "ended"},
        },
    ]
    chosen = rulespec_parameter_version_at({"versions": versions}, "2013-01-01")
    assert chosen["values"] == {"0": "open"}


@pytest.mark.parametrize(
    "start, indexed_value",
    [("2019-12-31", "refused"), ("2020-01-01", {"kind": "integer", "value": 7})],
)
def test_indexed_parameter_keeps_the_period_rule(validator, start, indexed_value):
    parameter = {
        "id": f"{MODULE}#rate",
        "name": "rate",
        "indexed_by": "band",
        "versions": [
            {
                "effective_from": "2020-01-01",
                "values": {"2": {"kind": "integer", "value": 7}},
            }
        ],
    }
    inputs = {f"{MODULE}#input.band": 2}
    assert (
        _outcome(lambda: cli._rulespec_parameter_value(parameter, inputs, start))
        == indexed_value
    )
    assert (
        _outcome(
            lambda: validator._rulespec_compiled_parameter_value(
                parameter, {"start": start}, inputs
            )
        )
        == indexed_value
    )


@pytest.mark.parametrize("period", ENGINE_REFUSED_PERIODS)
def test_runners_refuse_periods_the_engine_cannot_read(validator, period):
    with pytest.raises(ValueError, match="period"):
        cli._rulespec_period_spec(period)
    with pytest.raises(ValueError, match="period"):
        validator._coerce_rulespec_period(period)


@pytest.mark.parametrize("name", [ABSENT, None, "", 2020, True, 1.5, ["day"]])
def test_runners_agree_a_custom_period_needs_a_text_name(validator, name):
    period = {"period_kind": "custom", "start": "2020-01-01", "end": "2020-12-31"}
    if name is not ABSENT:
        period["name"] = name
    with pytest.raises(ValueError, match="text name"):
        cli._rulespec_period_spec(period)
    with pytest.raises(ValueError, match="text name"):
        validator._coerce_rulespec_period(period)


def _period_forms() -> Iterator[dict]:
    for kind, start, end, name in product(
        PERIOD_KINDS, PERIOD_DATES, PERIOD_DATES, PERIOD_NAMES
    ):
        period = {"period_kind": kind, "start": start, "end": end}
        if name is not ABSENT:
            period["name"] = name
        yield period


def test_runners_accept_the_same_period_forms(validator):
    for period in _period_forms():
        outcomes = []
        for coerce in (cli._rulespec_period_spec, validator._coerce_rulespec_period):
            try:
                accepted = coerce(period)
            except ValueError:
                outcomes.append("refused")
                continue
            if accepted["period_kind"] != "custom":
                accepted.pop("name", None)
            outcomes.append(accepted)
        assert outcomes[0] == outcomes[1], period


def test_runners_read_a_bare_date_alike(validator):
    day = date(2020, 1, 1)
    assert cli._rulespec_period_spec(day) == validator._coerce_rulespec_period(day)


@given(number=st.integers(-(2**63), 2**63 - 1))
def test_a_whole_number_reads_as_its_index(number):
    assert rulespec_engine_index(number) == number
    if abs(number) < 2**53:
        assert rulespec_engine_index(float(number)) == number


@given(value=st.floats(allow_nan=True, allow_infinity=True))
def test_a_float_reads_as_an_index_only_as_plain_whole_text(value):
    """The engine reads a float's text, and that text has no exponent form."""
    plain = re.fullmatch(r"(-?\d+)\.0", str(value))
    assert rulespec_engine_index(value) == (int(plain[1]) if plain else None)


@pytest.mark.parametrize(
    "value, index",
    [
        (9999999999999998.0, 9999999999999998),
        (1e16, None),
        (-1e16, None),
        (1e300, None),
    ],
)
def test_a_float_index_stops_where_its_text_turns_to_an_exponent(value, index):
    assert rulespec_engine_index(value) == index


@pytest.mark.parametrize("number", [2**63, -(2**63) - 1, 2**64])
def test_a_number_past_64_bits_has_no_index(number):
    assert rulespec_engine_index(number) is None


@pytest.mark.parametrize("value", [True, False, "1", "one", None, [1], {"1": 1}])
def test_anything_else_has_no_index(value):
    assert rulespec_engine_index(value) is None


def _band_table(parameter_id: str | None = f"{MODULE}#rate") -> dict:
    """A compiled table with cells at 1 and 2, indexed by `band`."""
    table = {
        "name": "rate",
        "indexed_by": "band",
        "versions": [
            {
                "effective_from": "2020-01-01",
                "values": {
                    str(cell): {"kind": "integer", "value": cell} for cell in (1, 2)
                },
            }
        ],
    }
    return {**table, "id": parameter_id} if parameter_id is not None else table


@pytest.mark.parametrize("key", [item[0] for item in INDEX_KEYS.values() if item[1]])
def test_a_table_reads_its_index_from_the_engine_name_or_dummy(key):
    inputs = {key: 2, "us:statutes/other#input.band": 1}
    assert rulespec_index_input(inputs, _band_table()) == 2


@pytest.mark.parametrize(
    "key", [item[0] for item in INDEX_KEYS.values() if not item[1]]
)
def test_a_table_reads_no_index_from_any_other_key(validator, key):
    inputs = {key: 2}
    with pytest.raises(ValueError, match="needs input `band`"):
        cli._rulespec_parameter_value(_band_table(), inputs, "2020-01-01")
    with pytest.raises(ValueError, match="needs input `band`"):
        validator._rulespec_compiled_parameter_value(
            _band_table(), {"start": "2020-01-01"}, inputs
        )


@pytest.mark.parametrize("dummy_first", [True, False])
def test_the_engine_name_wins_over_the_dummy_key(dummy_first):
    pairs = [("dummy#input.band", 1), (f"{MODULE}#input.band", 2)]
    inputs = dict(pairs if dummy_first else pairs[::-1])
    assert rulespec_index_input(inputs, _band_table()) == 2


@pytest.mark.parametrize(
    "parameter_id, key",
    [
        (f"{MODULE}#rate", f"{MODULE}#input.band"),
        ("#rate", "#input.band"),
        (MODULE, "band"),
        (f"{MODULE}#", "band"),
        (None, "band"),
    ],
)
def test_a_table_takes_the_input_name_the_engine_gives_it(parameter_id, key):
    assert rulespec_index_input({key: 2}, _band_table(parameter_id)) == 2


def test_an_empty_index_name_still_needs_an_input():
    table = {**_band_table(), "indexed_by": ""}
    with pytest.raises(ValueError, match="needs input"):
        cli._rulespec_parameter_value(table, {}, "2020-01-01")


def test_a_missing_cell_refuses_instead_of_reading_cell_zero():
    table = _band_table()
    table["versions"][0]["values"]["0"] = {"kind": "integer", "value": 0}
    with pytest.raises(ValueError, match="has no value for key `3`"):
        cli._rulespec_parameter_value(table, {f"{MODULE}#input.band": 3}, "2020-01-01")


@pytest.mark.parametrize("text", ["20200101", "2020-W01-3", "2020-1-1", " 2020-01-01"])
def test_engine_date_reads_only_the_extended_form(text):
    with pytest.raises(ValueError, match="(?i)isoformat"):
        rulespec_engine_date(text)


def test_engine_date_reads_a_date_object_and_its_text():
    assert rulespec_engine_date(date(2020, 1, 1)) == date(2020, 1, 1)
    assert rulespec_engine_date("2020-01-01") == date(2020, 1, 1)


def test_parameter_only_case_fails_without_reaching_the_engine(tmp_path, monkeypatch):
    content_root = tmp_path / "rulespec-us" / "us"
    program = content_root / "statutes" / "example.yaml"
    program.parent.mkdir(parents=True)
    program.write_text("format: rulespec/v1\nrules: []\n")
    companion = program.with_name("example.test.yaml")
    companion.write_text(
        yaml.safe_dump(
            [
                {
                    "name": "before_the_first_version",
                    "period": _tax_year(2004),
                    "output": {f"{MODULE}#months": 6},
                },
                {
                    "name": "at_the_first_version",
                    "period": _tax_year(2005),
                    "output": {f"{MODULE}#months": 6},
                },
            ]
        )
    )
    artifact = {
        "program": {
            "parameters": [
                {
                    "id": f"{MODULE}#months",
                    "name": "months",
                    "versions": [
                        {
                            "effective_from": "2005-01-01",
                            "values": {"0": {"kind": "integer", "value": 6}},
                        }
                    ],
                }
            ]
        }
    }
    run = MagicMock()
    monkeypatch.setattr("axiom_encode.cli.subprocess.run", run)

    result = cli._execute_rulespec_test_file(
        companion,
        binary=tmp_path / "axiom-rules-engine",
        axiom_rules_path=tmp_path,
        env={},
        rulespec_roots=(tmp_path / "rulespec-us",),
        tmp_path=tmp_path,
        compiled_cache={program: (tmp_path / "compiled.json", artifact)},
        policy_repo_path=content_root,
    )

    assert result["failures"] == [
        {
            "file": str(companion),
            "case": "before_the_first_version",
            "message": (
                f"{MODULE}#months: parameter `months` has no version at 2004-01-01"
            ),
        }
    ]
    run.assert_not_called()


@pytest.fixture(scope="session")
def period_engine(real_engine) -> tuple[Path, Path]:
    """The real engine's binary and checkout (tests/real_engine.py)."""
    return real_engine.binary, real_engine.root


def _rule(name: str, dtype: str, versions: list[dict]) -> dict:
    return {"name": name, "kind": "parameter", "dtype": dtype, "versions": versions}


def _write_module(root: Path, rules: list[dict], cases: list[dict]) -> Path:
    module = root / "us" / "statutes" / "example.yaml"
    module.parent.mkdir(parents=True, exist_ok=True)
    module.write_text(yaml.safe_dump({"format": "rulespec/v1", "rules": rules}))
    module.with_suffix(".test.yaml").write_text(yaml.safe_dump(cases))
    return module


def _compile(engine: tuple[Path, Path], root: Path, module: Path, artifact: Path):
    binary, engine_root = engine
    result = run_rulespec_compile(
        binary=binary,
        program=module,
        rulespec_roots=[root],
        output=artifact,
        cwd=engine_root,
        env=os.environ.copy(),
    )
    assert result.returncode == 0, result.stderr or result.stdout
    return json.loads(artifact.read_text())


def _companion_failures(
    executor: str,
    engine: tuple[Path, Path],
    *,
    root: Path,
    module: Path,
    artifact: Path,
) -> list[str]:
    binary, engine_root = engine
    payload = _compile(engine, root, module, artifact)
    test_file = module.with_suffix(".test.yaml")
    cases = yaml.safe_load(test_file.read_text())
    if executor == "cli":
        result = cli._execute_rulespec_test_file(
            test_file,
            binary=binary,
            axiom_rules_path=engine_root,
            env=os.environ.copy(),
            rulespec_roots=[root],
            tmp_path=artifact.parent,
            compiled_cache={module: (artifact, payload)},
            policy_repo_path=root / "us",
        )
        assert result["cases"] == len(cases)
        return [
            f"Test case `{failure['case']}` {failure['message']}"
            for failure in result["failures"]
        ]
    return _pipeline(root / "us", engine_root)._run_rulespec_test_cases(
        rules_file=module,
        compiled_path=artifact,
        compiled_payload=payload,
        cases=cases,
    )


def _engine_query(
    engine: tuple[Path, Path], artifact: Path, outputs: list[str], period: dict
) -> subprocess.CompletedProcess[str]:
    binary, engine_root = engine
    return subprocess.run(
        [str(binary), "run-compiled", "--artifact", str(artifact)],
        input=json.dumps(
            {
                "mode": "explain",
                "dataset": {"inputs": [], "relations": []},
                "queries": [
                    {"entity_id": "case", "period": period, "outputs": outputs}
                ],
            }
        ),
        capture_output=True,
        text=True,
        timeout=60,
        cwd=str(engine_root),
        env=os.environ.copy(),
    )


@pytest.mark.parametrize("executor", ["cli", "validator"])
@pytest.mark.parametrize("name", SHAPES)
def test_assertion_needs_a_version_for_its_period(
    tmp_path, period_engine, executor, name
):
    dtype, formula, value, collection = SHAPES[name]
    versions = [
        {
            "effective_from": "2005-01-01",
            "effective_to": "2010-12-31",
            "formula": formula,
        }
    ]
    root = tmp_path / "rulespec-us"
    output = {f"{MODULE}#{name}": value}
    module = _write_module(
        root,
        [_rule(name, dtype, versions)],
        [
            {"name": "first", "period": _tax_year(2005), "output": output},
            {"name": "last", "period": _tax_year(2010), "output": output},
        ],
    )
    artifact = tmp_path / "compiled.json"
    payload = _compile(period_engine, root, module, artifact)
    assert [item["name"] for item in payload["program"][collection]] == [name]
    assert (
        _companion_failures(
            executor, period_engine, root=root, module=module, artifact=artifact
        )
        == []
    )

    for year in (2004, 2011):
        _write_module(
            root,
            [_rule(name, dtype, versions)],
            [{"name": "outside", "period": _tax_year(year), "output": output}],
        )
        failures = _companion_failures(
            executor, period_engine, root=root, module=module, artifact=artifact
        )
        assert any(name in item and f"{year}-01-01" in item for item in failures)


@settings(
    max_examples=60,
    derandomize=True,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(versions=version_lists(min_size=1, loadable=True))
def test_runners_answer_exactly_the_periods_the_engine_answers(
    tmp_path, period_engine, versions
):
    """Each generated version layout, queried on every day around its bounds."""
    with tempfile.TemporaryDirectory(dir=tmp_path) as example_tmp:
        root = Path(example_tmp) / "rulespec-us"
        artifact = Path(example_tmp) / "compiled.json"
        rule_versions = [
            {
                **{key: item for key, item in version.items() if key != "values"},
                "formula": str(version["values"]["0"]["value"]),
            }
            for version in versions
        ]
        module = _write_module(root, [_rule("rate", "Integer", rule_versions)], [])
        program = _compile(period_engine, root, module, artifact)["program"]
        parameter = program["parameters"][0]
        validator = _pipeline(root / "us", period_engine[1])
        for day in [DAYS[0] - timedelta(days=1), *DAYS, DAYS[-1] + timedelta(days=1)]:
            period = _day_period(day)
            result = _engine_query(period_engine, artifact, [parameter["id"]], period)
            if result.returncode == 0:
                outputs = json.loads(result.stdout)["results"][0]["outputs"]
                engine_answer = outputs[parameter["id"]]["value"]
            else:
                assert "has no value for key" in result.stderr
                engine_answer = "refused"
            assert engine_answer == _outcome(
                lambda: cli._rulespec_parameter_value(parameter, {}, period["start"])
            )
            assert engine_answer == _outcome(
                lambda: validator._rulespec_compiled_parameter_value(parameter, period)
            )


def test_runners_answer_an_indexed_parameter_when_its_engine_lookup_does(
    tmp_path, period_engine
):
    """The engine refuses to query a table directly, so a derived rule reads it."""
    root = tmp_path / "rulespec-us"
    table = _rule(
        "rate_table",
        "Integer",
        [
            {
                "effective_from": DAYS[1].isoformat(),
                "effective_to": DAYS[3].isoformat(),
                "values": {1: 7},
            }
        ],
    )
    reader = {
        "name": "rate",
        "kind": "derived",
        "entity": "Person",
        "dtype": "Integer",
        "versions": [{"effective_from": "2000-01-01", "formula": "rate_table[1]"}],
    }
    module = _write_module(root, [{**table, "indexed_by": "band"}, reader], [])
    artifact = tmp_path / "compiled.json"
    program = _compile(period_engine, root, module, artifact)["program"]
    parameter = program["parameters"][0]
    output = program["derived"][0]["id"]
    validator = _pipeline(root / "us", period_engine[1])
    inputs = {f"{MODULE}#input.band": 1}
    answers = []
    for day in DAYS:
        period = _day_period(day)
        result = _engine_query(period_engine, artifact, [output], period)
        if result.returncode == 0:
            engine_answer = json.loads(result.stdout)["results"][0]["outputs"][output][
                "value"
            ]
        else:
            assert "has no value for key" in result.stderr
            engine_answer = "refused"
        assert engine_answer == _outcome(
            lambda: cli._rulespec_parameter_value(parameter, inputs, period["start"])
        )
        assert engine_answer == _outcome(
            lambda: validator._rulespec_compiled_parameter_value(
                parameter, period, inputs
            )
        )
        answers.append(engine_answer != "refused")
    assert answers == [False, True, True, True, False, False]


def _table_with_lookup(root: Path, cases: list[dict]) -> Path:
    table = _rule(
        "rate_table", "Integer", [{"effective_from": "2020-01-01", "values": TABLE}]
    )
    lookup = {
        "name": "rate",
        "kind": "derived",
        "entity": "Person",
        "dtype": "Integer",
        "versions": [{"effective_from": "2000-01-01", "formula": "rate_table[band]"}],
    }
    return _write_module(root, [{**table, "indexed_by": "band"}, lookup], cases)


@pytest.mark.parametrize("executor", ["cli", "validator"])
def test_a_direct_read_answers_only_the_index_the_engine_reads(
    tmp_path, period_engine, executor
):
    """The engine refuses to query a table, so its derived lookup is the witness."""
    root = tmp_path / "rulespec-us"
    cases = [
        {
            "name": f"{route}_{label}",
            "period": _tax_year(2020),
            "input": {f"{MODULE}#input.band": index},
            "output": {f"{MODULE}#{output}": 7 if cell is None else cell},
        }
        for label, (index, cell) in INDEXES.items()
        for route, output in (("direct", "rate_table"), ("lookup", "rate"))
    ]
    failures = _companion_failures(
        executor,
        period_engine,
        root=root,
        module=_table_with_lookup(root, cases),
        artifact=tmp_path / "compiled.json",
    )
    failed = set(re.findall(r"Test case `([^`]+)`", "\n".join(failures)))
    readable = {label for label, (_, cell) in INDEXES.items() if cell is not None}
    answered = {label for label in INDEXES if f"direct_{label}" not in failed}
    looked_up = {label for label in INDEXES if f"lookup_{label}" not in failed}
    assert answered == readable
    assert answered <= looked_up
    if executor == "cli":
        assert looked_up == readable
    # A wrong cell would fail too, so each failure must be the refusal itself.
    for label in set(INDEXES) - readable:
        refusal = "\n".join(item for item in failures if f"`direct_{label}`" in item)
        assert "must be an integer" in refusal or "has no value for key" in refusal


@pytest.mark.parametrize("executor", ["cli", "validator"])
def test_only_dummy_answers_a_key_the_engine_refuses(tmp_path, period_engine, executor):
    """Dummy is the one key a direct read takes and the engine's lookup refuses."""
    root = tmp_path / "rulespec-us"
    cases = [
        {
            "name": f"{route}_{label}",
            "period": _tax_year(2020),
            "input": {key: 1},
            "output": {f"{MODULE}#{output}": 7},
        }
        for label, (key, _) in INDEX_KEYS.items()
        for route, output in (("direct", "rate_table"), ("lookup", "rate"))
    ]
    failures = _companion_failures(
        executor,
        period_engine,
        root=root,
        module=_table_with_lookup(root, cases),
        artifact=tmp_path / "compiled.json",
    )
    failed = set(re.findall(r"Test case `([^`]+)`", "\n".join(failures)))
    direct = {label for label in INDEX_KEYS if f"direct_{label}" not in failed}
    lookup = {label for label in INDEX_KEYS if f"lookup_{label}" not in failed}
    assert direct == {"engine_name", "dummy"}
    assert lookup == {"engine_name"}


@pytest.mark.parametrize("executor", ["cli", "validator"])
@pytest.mark.parametrize("bands", [(1, 2), (2, 1)], ids=["1_then_2", "2_then_1"])
def test_a_table_row_supplies_no_index_for_a_direct_read(
    tmp_path, period_engine, executor, bands
):
    """Two rows could read two cells, so neither row answers a direct assertion."""
    root = tmp_path / "rulespec-us"
    rows = [
        {"id": f"person{row}", f"{MODULE}#input.band": band}
        for row, band in enumerate(bands)
    ]
    case = {
        "name": "rows",
        "period": _tax_year(2020),
        "tables": {"Person": rows},
        "output": {f"{MODULE}#rate_table": 7},
    }
    failures = _companion_failures(
        executor,
        period_engine,
        root=root,
        module=_table_with_lookup(root, [case]),
        artifact=tmp_path / "compiled.json",
    )
    assert any("needs input `band`" in item for item in failures), failures


def test_tied_versions_read_what_the_engine_reads(tmp_path, period_engine):
    """The compiler refuses a tie, so this edits the compiled artifact by hand."""
    root = tmp_path / "rulespec-us"
    versions = [{"effective_from": "2005-01-01", "formula": "6"}]
    module = _write_module(root, [_rule("months", "Integer", versions)], [])
    artifact = tmp_path / "compiled.json"
    payload = _compile(period_engine, root, module, artifact)
    parameter = payload["program"]["parameters"][0]
    tied = copy.deepcopy(parameter["versions"][0])
    tied["values"]["0"]["value"] = 7
    parameter["versions"].append(tied)
    artifact.write_text(json.dumps(payload))
    result = _engine_query(period_engine, artifact, [parameter["id"]], _tax_year(2020))
    assert result.returncode == 0, result.stderr
    outputs = json.loads(result.stdout)["results"][0]["outputs"]
    answer = cli._rulespec_parameter_value(parameter, {}, "2020-01-01")
    assert answer == outputs[parameter["id"]]["value"] == tied["values"]["0"]


def _months_artifact(root: Path, engine: tuple[Path, Path]) -> Path:
    versions = [{"effective_from": "2005-01-01", "formula": "6"}]
    module = _write_module(root, [_rule("months", "Integer", versions)], [])
    artifact = root / "compiled.json"
    _compile(engine, root, module, artifact)
    return artifact


@pytest.mark.parametrize("period", ENGINE_REFUSED_PERIODS)
def test_the_engine_refuses_the_periods_the_runners_refuse(
    tmp_path, period_engine, period
):
    artifact = _months_artifact(tmp_path / "rulespec-us", period_engine)
    assert _engine_query(period_engine, artifact, [], _tax_year(2020)).returncode == 0
    assert _engine_query(period_engine, artifact, [], period).returncode != 0


def test_the_engine_reads_every_period_a_runner_accepts(
    tmp_path, period_engine, validator
):
    """Crosses every period form; only accepted ones reach the engine."""
    artifact = _months_artifact(tmp_path / "rulespec-us", period_engine)
    accepted = set()
    for period in _period_forms():
        for coerce in (cli._rulespec_period_spec, validator._coerce_rulespec_period):
            try:
                accepted.add(json.dumps(coerce(period), sort_keys=True))
            except ValueError:
                continue
    kinds = {json.loads(text)["period_kind"] for text in accepted}
    assert kinds == {"month", "benefit_week", "tax_year", "custom"}
    for text in sorted(accepted):
        result = _engine_query(period_engine, artifact, [], json.loads(text))
        assert result.returncode == 0, (text, result.stderr)
