"""Closed source-owned separate-country calculation requirement evidence.

No generic income polarity is relaxed. Only an exact typed and executed
source operation can provide narrowly scoped condition evidence.
"""

from __future__ import annotations

import ast
import hashlib
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import yaml

from axiom_encode.harness import source_completeness as sc
from axiom_encode.harness import source_path_evidence as pe
from axiom_encode.harness import worksheet_operation_evidence as op
from axiom_encode.harness.proof_validator import validate_rulespec_proofs

OWNER = "separate_country_forms_required"
COUNT = "foreign_countries_taxed_count"
AMOUNT = "total_non_business_income_taxes_paid_to_all_foreign_countries"
LIMIT = AMOUNT + "_ceiling"
CITE = "ca/policy/cra/t1-2025/provincial-territorial-foreign-tax-credit"
CLAUSE = (
    "If you paid tax to more than one foreign country, and the total of the non-business income taxes that you paid to all foreign\n"
    "countries was more than $200, do the calculation on a separate form for each country in Canadian dollars."
)
ANTECEDENT = CLAUSE.split(", do the calculation", 1)[0]
PREDECESSOR = "As a result, you do not have to complete this form.\n"
PRECEDING_CONTEXT = "\nThis form does not apply to residents of Quebec.\nIf you must pay minimum tax and you were a resident of Manitoba, you cannot claim a provincial foreign tax credit.\nBefore you complete this form, calculate your federal foreign tax credit by using Form T2209, Federal Foreign Tax Credits. If\nthe amount of the federal foreign non-business income tax credit you are entitled to deduct is equal to the foreign non-business\ntax you paid, your provincial or territorial foreign tax credit would be zero. As a result, you do not have to complete this form.\n"
DESCRIPTIONS = {
    COUNT: "number of foreign countries to which non-business income tax was paid",
    AMOUNT: "total non-business income taxes paid to all foreign countries",
}


@dataclass(frozen=True)
class CountryGateCoverage:
    unresolved: tuple[str, ...]
    identity: str = ""
    source_hash: str = ""
    owner: str = OWNER
    version: int = 0
    span: tuple[int, int] | None = None
    interval: tuple[str, str] | None = None
    pairs: tuple[tuple[str, int, int], ...] = ()

    def owns_condition(
        self,
        *,
        payload: Any,
        test_cases: Any,
        source_text: Any,
        corpus_citation_path: str,
        owner: str,
        version: int,
        interval_start: str | None,
        interval_end: str | None,
        start: int,
        end: int,
    ) -> bool:
        """Bind exact clause, effective version and immutable analysis inputs."""
        if self.unresolved or self.span is None or self.interval is None:
            return False
        if (
            corpus_citation_path != CITE
            or not isinstance(payload, dict)
            or not isinstance(test_cases, list)
            or len(test_cases) > 64
            or not isinstance(source_text, str)
        ):
            return False
        a, b = self.span
        if (
            not (0 <= a <= b <= len(source_text))
            or self.source_hash != hashlib.sha256(source_text.encode()).hexdigest()
        ):
            return False
        if any(type(value) is not int for value in (version, start, end)):
            return False
        whitespace_start = a
        while whitespace_start and source_text[whitespace_start - 1].isspace():
            whitespace_start -= 1
        if not (
            owner == self.owner
            and version == self.version
            and (interval_start, interval_end) == self.interval
            and start in {a, whitespace_start}
            and end == b
        ):
            return False
        try:
            rules = {rule["name"]: rule for rule in payload["rules"]}
            if len(rules) != len(payload["rules"]):
                return False
            if any(
                not isinstance(case, dict)
                or not isinstance(case.get("input"), dict)
                or not isinstance(case.get("output"), dict)
                for case in test_cases
            ):
                return False
            normalized = sc._typed_date_cases(
                sc._typed_numeric_expected_cases(test_cases, rules), payload
            )
            if (
                self.identity != pe.path_input_identity(payload, normalized)
                or self.source_hash != hashlib.sha256(source_text.encode()).hexdigest()
            ):
                return False
        except (
            TypeError,
            ValueError,
            KeyError,
            AttributeError,
            RecursionError,
            OverflowError,
        ):
            return False
        return True


def _certify(
    payload: dict[str, Any], cases: list[dict[str, Any]], source: str
) -> CountryGateCoverage:
    def fail(reason):
        return CountryGateCoverage((reason,))

    rules = payload.get("rules", [])
    inputs = payload.get("inputs", [])
    if not isinstance(rules, list) or not isinstance(inputs, list):
        return fail("malformed declarations")
    if any(not isinstance(x, dict) for x in rules + inputs):
        return fail("malformed declaration")
    r = {x.get("name"): x for x in rules}
    facts = {x.get("name"): x for x in inputs}
    if len(r) != len(rules) or len(facts) != len(inputs) or set(r) & set(facts):
        return fail("ambiguous declarations")
    if OWNER not in r or LIMIT not in r or not {COUNT, AMOUNT} <= facts.keys():
        return fail("unbound input/output identity")
    if payload.get("imports"):
        return fail("imported contract unresolved")
    sha = hashlib.sha256(source.encode()).hexdigest()
    if payload.get("module", {}).get("source_verification") != {
        "corpus_citation_path": CITE,
        "source_sha256": sha,
    }:
        return fail("source identity")
    quotes = sc._annual_source_quoted_spans(source)
    if quotes is None or source.count(CLAUSE) != 1:
        return fail("unowned source clause")
    start = source.index(CLAUSE)
    end = start + len(CLAUSE)
    intro = op._annual_intro(source)
    if intro is None or source[intro.end() : start] != PRECEDING_CONTEXT:
        return fail("unknown preceding restriction/context")
    if any(a < end and b > start for a, b in quotes):
        return fail("quoted source clause")
    # Closed following printed field ensures no hidden continuation/restriction.
    if not source[end:].startswith(
        "\nCountry or countries for which you are making this claim:\n"
    ):
        return fail("unconsumed source restriction")
    intro = op._annual_intro(source)
    if intro is None:
        return fail("unowned annual source")
    year = intro["year"]
    interval = (year + "-01-01", year + "-12-31")
    prefix = sc._collapse_text(intro.group().split(" that you can deduct", 1)[0])
    for name, dtype, unit in ((COUNT, "Integer", None), (AMOUNT, "Money", "CAD")):
        item = facts[name]
        if (
            item.get("entity"),
            item.get("period"),
            item.get("dtype"),
            item.get("unit"),
        ) != ("Person", "Year", dtype, unit):
            return fail("input type/entity/period/unit")
        # Exact finite identity contract; unknown descriptions are unresolved,
        # never used to relabel a has-income/no-income or payable-tax fact.
        desc = item.get("description", "")
        if not isinstance(desc, str) or (
            desc and sc._collapse_text(desc).lower() != DESCRIPTIONS[name]
        ):
            return fail("input description not owned")
    owner = r[OWNER]
    parameter = r[LIMIT]
    if (
        owner.get("kind"),
        owner.get("entity"),
        owner.get("period"),
        owner.get("dtype"),
    ) not in [
        ("derived", "Person", "Year", "Judgment"),
        ("derived", "Person", "Year", "Boolean"),
    ]:
        return fail("owner type or polarity")
    if owner.get("unit") is not None:
        return fail("owner unit")
    if owner.get("description"):
        return fail("owner description unresolved")
    if (parameter.get("kind"), parameter.get("dtype"), parameter.get("unit")) != (
        "parameter",
        "Money",
        "CAD",
    ):
        return fail("threshold type")
    if parameter.get("entity") is not None or parameter.get("period") is not None:
        return fail("parameter context unsupported")
    selected = {}
    for name in (OWNER, LIMIT):
        intervals = sc._effective_formula_version_intervals(r[name])
        if len(intervals) != 1 or intervals[0][2:] != interval:
            return fail("selected effective interval")
        index = intervals[0][0]
        selected[name] = index
        atoms = sc._rule_source_excerpt_atoms(r[name])
        for field in ("effective_from", "effective_to"):
            if not any(
                p == f"versions[{index}].{field}"
                and c == CITE
                and prefix in sc._collapse_text(ex)
                and sc._collapse_text(ex) in sc._collapse_text(source)
                for p, c, ex in atoms
            ):
                return fail("selected date ownership")
    i = selected[OWNER]
    j = selected[LIMIT]
    formula = owner["versions"][i].get("formula")
    threshold = parameter["versions"][j].get("formula")
    if type(threshold) not in (int, float) or threshold != 200:
        return fail("wrong source threshold")
    expected = ast.parse(f"{COUNT} > 1 and {AMOUNT} > {LIMIT}", mode="eval").body
    actual = sc._parse_formula_expression(formula) if isinstance(formula, str) else None
    if actual is None or ast.dump(actual) != ast.dump(expected):
        return fail("source predicate differs")
    atoms = sc._rule_source_excerpt_atoms(owner)
    if not any(
        p == f"versions[{i}].formula"
        and c == CITE
        and sc._collapse_text(CLAUSE) == sc._collapse_text(ex)
        for p, c, ex in atoms
    ):
        return fail("owner lacks complete consequence proof")
    subset = {**payload, "rules": [owner, parameter]}
    # Interpretations refer to the full module, not this temporary proof view.
    # The shared validation gate checks them against the original declarations.
    if isinstance(subset.get("module"), Mapping):
        subset["module"] = {
            key: value
            for key, value in subset["module"].items()
            if key != "interpretations"
        }
    proof = validate_rulespec_proofs(
        yaml.safe_dump(subset),
        require_policy_proofs=True,
        source_texts={CITE: source},
    )
    if not proof.passed:
        return fail("normal selected proof validation: " + str(proof.issues))
    if not any(
        p == f"versions[{j}].formula"
        and c == CITE
        and sc._collapse_text(ex)
        in {sc._collapse_text(CLAUSE), sc._collapse_text(ANTECEDENT)}
        for p, c, ex in sc._rule_source_excerpt_atoms(parameter)
    ):
        return fail("threshold source provenance")
    normalized = sc._typed_date_cases(
        sc._typed_numeric_expected_cases(cases, r), payload
    )
    links = []
    expected_prefix = (
        "ca:policies/cra/t1-2025/provincial-territorial-foreign-tax-credit#"
    )
    for n, case in enumerate(normalized):
        if not isinstance(case, dict):
            return fail("malformed case")
        if set(case) - {"name", "period", "input", "output"}:
            return fail("unsupported case entity/context")
        outputs = case.get("output", {})
        keys = [k for k in outputs if k.rsplit("#", 1)[-1] == OWNER]
        if not keys:
            continue
        if keys != [expected_prefix + OWNER]:
            return fail("foreign output identity")
        period = case.get("period")
        if period != {
            "period_kind": "tax_year",
            "start": interval[0],
            "end": interval[1],
        }:
            return fail("case period")
        env = sc._case_input_formula_environment(case)
        if env is None or not {COUNT, AMOUNT} <= env.keys():
            return fail("missing reached facts")
        for name in (COUNT, AMOUNT):
            if expected_prefix + "input." + name not in case["input"] or any(
                k.endswith("input." + name) and k != expected_prefix + "input." + name
                for k in case["input"]
            ):
                return fail("foreign input identity")
        if type(env[COUNT]) is not int or not -(2**63) <= env[COUNT] < 2**63:
            return fail("count domain")
        if type(env[AMOUNT]) not in (int, float) or not math.isfinite(env[AMOUNT]):
            return fail("amount domain")
        execution = sc._case_formula_execution(
            owner, case, formula_environment={LIMIT: 200}
        )
        if execution is None:
            return fail("unresolved owner execution")
        value = sc._formula_execution_runtime_value(execution)
        asserted = sc._test_case_asserted_output_value(case, OWNER)
        if not sc._asserted_formula_runtime_values_equal(owner, value, asserted):
            return fail("uncorroborated owner assertion")
        links.append((n, case, value, env))
    pairs = []
    for name in (COUNT, AMOUNT):
        found = []
        for a in links:
            for b in links:
                if a[2] is False and b[2] is True and pe._same_world(a[1], b[1], name):
                    found.append((name, a[0], b[0]))
        if not found:
            return fail("missing same-world predicate pair: " + name)
        pairs.append(found[0])
    return CountryGateCoverage(
        (),
        pe.path_input_identity(payload, normalized),
        sha,
        OWNER,
        i,
        (start, end),
        interval,
        tuple(pairs),
    )


def _certify_checked(payload: Any, cases: Any, source: Any) -> CountryGateCoverage:
    """Bounded malformed-data rejection before existing diagnostic adapters."""
    if (
        not isinstance(payload, dict)
        or not isinstance(source, str)
        or not isinstance(cases, list)
    ):
        return CountryGateCoverage(("malformed top-level contract",))
    if len(cases) > 64:
        return CountryGateCoverage(("case bound exceeded",))
    for case in cases:
        if (
            not isinstance(case, dict)
            or not isinstance(case.get("input"), dict)
            or not isinstance(case.get("output"), dict)
        ):
            return CountryGateCoverage(("malformed case/input/output mapping",))
        if any(
            not isinstance(k, str) for field in ("input", "output") for k in case[field]
        ):
            return CountryGateCoverage(("malformed qualified key",))
    try:
        return _certify(payload, cases, source)
    except (
        TypeError,
        ValueError,
        KeyError,
        AttributeError,
        RecursionError,
        OverflowError,
        yaml.YAMLError,
    ) as error:
        return CountryGateCoverage(
            ("malformed unresolved contract: " + type(error).__name__,)
        )


def certify_worksheet_country(
    payload: Any,
    *,
    test_cases: Any,
    source_text: Any,
    corpus_citation_path: str,
) -> CountryGateCoverage:
    """Return only exact source/proof/date/type/execution-backed evidence."""
    if corpus_citation_path != CITE:
        return CountryGateCoverage(("different source citation",))
    return _certify_checked(payload, test_cases, source_text)
