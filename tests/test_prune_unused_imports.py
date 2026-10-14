"""Deterministic unused-import pruning keeps `imports` a valid sequence.

us/statute/26/219 (run 35863402410) failed compile with "imports: invalid
type: unit value, expected a sequence" after pruning removed every import and
left a bare `imports:` header, which loads as null.
"""

from __future__ import annotations

import pytest
import yaml

from axiom_encode import cli
from axiom_encode.harness import evals

_ALL_UNUSED = """format: rulespec/v1
imports:
- us:statutes/26/1/h#net_capital_gain
- us:statutes/26/1/h#qualified_dividend_income
module:
  summary: Example.
rules:
- name: deductible_limit
  kind: parameter
  dtype: Money
  versions:
  - effective_from: '2026-01-01'
    formula: '7500'
"""

_ONE_USED = """format: rulespec/v1
imports:
  - us:statutes/26/1/h#net_capital_gain
  - us:statutes/26/1/h#qualified_dividend_income
rules:
  - name: copied_gain
    kind: derived
    dtype: Money
    period: Year
    versions:
      - effective_from: '2026-01-01'
        formula: net_capital_gain
"""


@pytest.mark.parametrize(
    "prune", [evals._prune_unused_imports, cli._prune_unused_imports]
)
def test_pruning_every_import_drops_the_imports_header(prune):
    repaired, removed = prune(_ALL_UNUSED)

    assert removed == [
        "us:statutes/26/1/h#net_capital_gain",
        "us:statutes/26/1/h#qualified_dividend_income",
    ]
    payload = yaml.safe_load(repaired)
    assert "imports" not in payload
    assert payload["module"] == {"summary": "Example."}
    assert payload["rules"][0]["name"] == "deductible_limit"


@pytest.mark.parametrize(
    "prune", [evals._prune_unused_imports, cli._prune_unused_imports]
)
def test_pruning_keeps_the_header_while_an_import_remains(prune):
    repaired, removed = prune(_ONE_USED)

    assert removed == ["us:statutes/26/1/h#qualified_dividend_income"]
    assert yaml.safe_load(repaired)["imports"] == [
        "us:statutes/26/1/h#net_capital_gain"
    ]


@pytest.mark.parametrize(
    "prune", [evals._prune_unused_imports, cli._prune_unused_imports]
)
def test_pruning_without_unused_imports_leaves_content_unchanged(prune):
    content = _ONE_USED.replace(
        "        formula: net_capital_gain",
        "        formula: net_capital_gain + qualified_dividend_income",
    )

    assert prune(content) == (content, [])
