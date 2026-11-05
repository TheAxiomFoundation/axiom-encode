"""Ordered local row ownership, never evidence of eligibility or factual truth.

An entered row is an explicit rule-level declaration of an opaque fact. The
caller still needs ordinary same-case execution evidence; this gate grants no
witness and cannot authenticate a simultaneous relabelling of all input facts.
"""

from __future__ import annotations

import ast
import hashlib
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import PurePosixPath
from typing import Any, Mapping

from axiom_encode.bound_form_structure import BoundFormStructure

from .bound_form_obligations import (
    DocumentRequirement,
    bind_document_requirements,
    observed_document_pairs,
)
from .imported_export_evidence import _atoms, _named, _payload

_LABEL = r"[A-Za-z][A-Za-z ,'-]*"
_FORBIDDEN = re.compile(r"\b(?:if|unless|except|provided|otherwise|when)\b", re.I)
_OPERATIVE_LABEL = re.compile(
    r"\b(?:plus|minus|less|subtract|add|multiply|divide|maximum|minimum|reduced|increased|after|before|including|excluding|deduct|reduce|increase|times)\b|\bnet of\b",
    re.I,
)
_SUB = re.compile(
    r"Line (?P<a>[1-9]\d*) minus line (?P<b>[1-9]\d*)(?: \(maximum \$(?P<cap>\d[\d,]*(?:\.\d+)?)\))?(?: (?P<label>"
    + _LABEL
    + r"))? = \^"
)
_MUL = re.compile(
    r"Line (?P<a>[1-9]\d*) multiplied by the percentage from line (?P<b>[1-9]\d*) (?P<label>"
    + _LABEL
    + r") = \^"
)
_ADD = re.compile(
    r"Line (?P<a>[1-9]\d*) plus line (?P<b>[1-9]\d*) Enter this amount on\s*line [1-9]\d*\s*of your return\. (?P<label>"
    + _LABEL
    + r") = \^"
)
_RATE = re.compile(r"Applicable rate (?P<rate>\d+(?:\.\d+)?)%")
_ENTERED = re.compile(
    r"(?:Total amount paid from all receipts included in the table above Total eligible expenses = \^"
    r"|Amount of assistance received or receivable included in the amount on line [1-9]\d* \^"
    r"|Amount from line [1-9]\d* being claimed by other eligible individuals \^"
    r"|Enter the [A-Za-z ]+ amount from line [1-9]\d* of your Form [A-Z][A-Z0-9]*\. \^)"
)


def _tree(value: Any) -> ast.expr | None:
    if type(value) not in (str, int, float):
        return None
    try:
        return ast.parse(str(value).strip(), mode="eval").body
    except (SyntaxError, ValueError, RecursionError):
        return None


def _number(value: Any) -> Decimal | None:
    node = _tree(value)
    if not isinstance(node, ast.Constant) or type(node.value) not in (int, float):
        return None
    try:
        number = Decimal(str(node.value))
        return number if number.is_finite() else None
    except InvalidOperation:
        return None


def _proof(
    rule: Mapping[str, Any], path: str, kind: str, text: str, form: BoundFormStructure
) -> bool:
    atoms = _atoms(rule, path, kind)
    if len(atoms) != 1:
        return False
    # Optional unverified span/hash fields are not accepted by this certificate.
    return (
        atoms[0].get("source")
        == {"corpus_citation_path": form.citation, "excerpt": text}
        and form.source.count(text) == 1
    )


@dataclass(frozen=True)
class OrderedFormBinding:
    form: BoundFormStructure
    content_sha256: str
    row_owners: tuple[tuple[int, str], ...]
    formulas: tuple[tuple[str, str], ...]
    entered: tuple[tuple[str, str], ...]
    entity: str
    unit: str
    consumer_namespace: str | None
    document_requirements: tuple[DocumentRequirement, ...] = ()

    def matches(
        self,
        *,
        content: str,
        branch: Any,
        rule: Mapping[str, Any] | None,
        principal_rules: Mapping[str, Any] | None,
        case: Mapping[str, Any] | None,
        leaf: str,
        trace: Any,
    ) -> bool:
        if (
            hashlib.sha256(content.encode()).hexdigest() != self.content_sha256
            or rule is None
            or case is None
            or principal_rules is None
            or trace
        ):
            return False
        payload = _payload(content)
        rules = _named(payload, "rules") if payload else None
        if rules is None or any(rules.get(k) != v for k, v in principal_rules.items()):
            return False
        rows = {r.number: r for r in self.form.rows}
        owners = dict(self.row_owners)
        matching = [
            n
            for n, r in rows.items()
            if r.span == (branch.start, branch.end)
            and self.form.source[slice(*r.span)] == branch.text
        ]
        if (
            len(matching) != 1
            or owners.get(matching[0]) != rule.get("name")
            or rules.get(rule.get("name")) != rule
        ):
            return False
        expected = dict(self.formulas).get(rule.get("name"))
        actual = _tree(leaf)
        if expected is None or actual is None or ast.dump(actual) != expected:
            return False
        period = case.get("period")
        year = str(self.form.header.year)
        if isinstance(period, dict):
            if period != {
                "period_kind": "tax_year",
                "start": year + "-01-01",
                "end": year + "-12-31",
            }:
                return False
        elif period not in (year, year + "-01-01"):
            return False
        facts = case.get("input")
        if not isinstance(facts, dict):
            return False
        aliases = []
        for key in facts:
            if not isinstance(key, str):
                return False
            if "#" in key:
                if self.consumer_namespace is None or not key.startswith(
                    self.consumer_namespace + "#input."
                ):
                    return False
                alias = key[len(self.consumer_namespace + "#input.") :]
            else:
                alias = key.removeprefix("input.")
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", alias):
                return False
            aliases.append(alias)
        if len(aliases) != len(set(aliases)) or set(aliases) & set(rules):
            return False
        values = dict(zip(aliases, facts.values()))
        # No omitted fact, default, expected-output value, or other-case bridge.
        required = set()
        pending = [rule["name"]]
        visited = set()
        entered = dict(self.entered)
        while pending:
            name = pending.pop()
            if name in visited:
                continue
            visited.add(name)
            if name in entered:
                required.add(entered[name])
            else:
                node = _tree(rules[name]["versions"][0]["formula"])
                pending.extend(
                    n.id
                    for n in ast.walk(node)
                    if isinstance(n, ast.Name) and n.id in rules
                )
        return all(
            name in values and _number(values[name]) is not None for name in required
        )


def bind_ordered_form(
    form: BoundFormStructure,
    content: str,
    proof_sources: Mapping[str, str | None],
    *,
    root_relative: str | None = None,
) -> OrderedFormBinding | None:
    """Certify the graph; root_relative comes only from bound execution evidence."""
    namespace = None
    if root_relative is not None:
        if (
            not isinstance(root_relative, str)
            or re.fullmatch(
                r"[a-z][a-z0-9-]*/(?:policies|statutes|regulations)/(?:[A-Za-z0-9_-]+/)*[A-Za-z0-9_-]+\.yaml",
                root_relative,
            )
            is None
        ):
            return None
        relative = PurePosixPath(root_relative)
        namespace = f"{relative.parts[0]}:{PurePosixPath(*relative.parts[1:]).with_suffix('').as_posix()}"
    if (
        not form.bound(form.source, form.citation)
        or proof_sources.get(form.citation) != form.source
    ):
        return None
    payload = _payload(content)
    if payload is None or payload.get("imports", []) != []:
        return None
    rules, inputs, units = (
        _named(payload, key) for key in ("rules", "inputs", "units")
    )
    if rules is None or inputs is None or units is None or set(rules) & set(inputs):
        return None
    module = payload.get("module")
    verification = (
        module.get("source_verification") if isinstance(module, dict) else None
    )
    if (
        not isinstance(verification, dict)
        or verification.get("corpus_citation_path") != form.citation
        or verification.get("source_sha256") != form.source_sha256
    ):
        return None
    # Classify every original declaration; unknown kinds still hit the closed
    # row grammar or the closed document grammar, never an ignored-extra path.
    document_rules = {n: r for n, r in rules.items() if r.get("dtype") == "Judgment"}
    document_inputs = set()
    for rule in document_rules.values():
        versions = rule.get("versions")
        if (
            not isinstance(versions, list)
            or len(versions) != 1
            or not isinstance(versions[0], dict)
        ):
            return None
        expression = _tree(versions[0].get("formula"))
        if not isinstance(expression, ast.Name) or expression.id not in inputs:
            return None
        document_inputs.add(expression.id)
    documents = bind_document_requirements(
        form, document_rules, {n: inputs[n] for n in document_inputs}
    )
    if documents is None or payload.get("relations", []) != []:
        return None
    rules = {n: r for n, r in rules.items() if n not in document_rules}
    inputs = {n: r for n, r in inputs.items() if n not in document_inputs}
    row_text = {row.number: form.source[slice(*row.span)] for row in form.rows}
    specs = {}
    for row in form.rows:
        caption = " ".join(form.source[slice(*row.caption_span)].split())
        if _FORBIDDEN.search(caption):
            return None
        found = [
            (kind, pattern.fullmatch(caption))
            for kind, pattern in (
                ("sub", _SUB),
                ("mul", _MUL),
                ("add", _ADD),
                ("rate", _RATE),
                ("entered", _ENTERED),
            )
        ]
        found = [(kind, match) for kind, match in found if match is not None]
        if len(found) != 1:
            return None
        label = found[0][1].groupdict().get("label")
        if label and _OPERATIVE_LABEL.search(label):
            return None
        specs[row.number] = found[0]
    owners, cap_owners, formulas, entered = {}, {}, {}, {}
    frame = (f"{form.header.year}-01-01", f"{form.header.year}-12-31")
    dimensions = set()
    for name, rule in rules.items():
        metadata = rule.get("metadata")
        proof = metadata.get("proof") if isinstance(metadata, dict) else None
        atoms = proof.get("atoms") if isinstance(proof, dict) else None
        expected_atoms = {
            ("versions[0].formula", "formula"),
            ("versions[0].effective_from", "effective_period"),
            ("versions[0].effective_to", "effective_period"),
        }
        if (
            not isinstance(atoms, list)
            or len(atoms) != 3
            or any(
                not isinstance(a, dict)
                or set(a) != {"path", "kind", "source"}
                or not isinstance(a.get("path"), str)
                or not isinstance(a.get("kind"), str)
                for a in atoms
            )
            or {(a["path"], a["kind"]) for a in atoms} != expected_atoms
        ):
            return None
        versions = rule.get("versions")
        # One annual source header cannot certify an additional date interval.
        if (
            not isinstance(versions, list)
            or len(versions) != 1
            or not isinstance(versions[0], dict)
        ):
            return None
        version = versions[0]
        if (
            set(version) != {"formula", "effective_from", "effective_to"}
            or tuple(version[k] for k in ("effective_from", "effective_to")) != frame
        ):
            return None
        if any(
            not _proof(
                rule,
                "versions[0]." + k,
                "effective_period",
                str(form.header.year),
                form,
            )
            for k in ("effective_from", "effective_to")
        ):
            return None
        matches = [
            n
            for n, text in row_text.items()
            if _proof(rule, "versions[0].formula", "formula", text, form)
        ]
        if len(matches) != 1:
            return None
        n = matches[0]
        kind, source = specs[n]
        formula = _tree(version["formula"])
        if formula is None:
            return None
        if rule.get("kind") == "parameter":
            amount = _number(version["formula"])
            if (
                kind == "rate"
                and rule.get("dtype") == "Rate"
                and amount == Decimal(source["rate"]) / 100
            ):
                destination = owners
            elif (
                kind == "sub"
                and source["cap"]
                and rule.get("dtype") == "Money"
                and amount == Decimal(source["cap"].replace(",", ""))
            ):
                destination = cap_owners
            else:
                return None
        elif (
            rule.get("kind") == "derived"
            and rule.get("dtype") == "Money"
            and rule.get("period") == "Year"
            and isinstance(rule.get("entity"), str)
        ):
            dimensions.add((rule["entity"], rule.get("unit")))
            destination = owners
            if kind == "entered":
                if not isinstance(formula, ast.Name) or formula.id not in inputs:
                    return None
                fact = inputs[formula.id]
                if (
                    any(
                        fact.get(k) != rule.get(k)
                        for k in ("dtype", "unit", "entity", "period")
                    )
                    or formula.id in entered.values()
                ):
                    return None
                entered[name] = formula.id
        else:
            return None
        if n in destination:
            return None
        destination[n] = name
        formulas[name] = ast.dump(formula)
    if (
        set(owners) != set(specs)
        or len(dimensions) != 1
        or set(entered.values()) != set(inputs)
    ):
        return None
    entity, unit = dimensions.pop()
    if units.get(unit, {}).get("kind") != "currency" or any(
        rules[name].get("unit") != unit for name in cap_owners.values()
    ):
        return None
    for n, (kind, source) in specs.items():
        if kind in ("rate", "entered"):
            continue
        a, b = int(source["a"]), int(source["b"])
        if a == b or a not in owners or b not in owners or a >= n or b >= n:
            return None
        if kind == "mul" and specs[b][0] != "rate":
            return None
        if kind != "mul" and (specs[a][0] == "rate" or specs[b][0] == "rate"):
            return None
        op = {"sub": ast.Sub, "mul": ast.Mult, "add": ast.Add}[kind]()
        expected = ast.BinOp(
            ast.Name(owners[a], ast.Load()), op, ast.Name(owners[b], ast.Load())
        )
        if kind == "sub" and source["cap"]:
            if n not in cap_owners:
                return None
            expected = ast.Call(
                ast.Name("min", ast.Load()),
                [ast.Name(cap_owners[n], ast.Load()), expected],
                [],
            )
        if formulas[owners[n]] != ast.dump(expected):
            return None
    return OrderedFormBinding(
        form,
        hashlib.sha256(content.encode()).hexdigest(),
        tuple(owners.items()),
        tuple(formulas.items()),
        tuple(entered.items()),
        entity,
        unit,
        namespace,
        documents,
    )


def bind_local_form_snapshot(
    content: str, sources: Mapping[str, str | None], root_relative: str
) -> bool:
    """Closed no-import eligibility; capture still enforces Git/engine freshness."""
    from axiom_encode.bound_form_structure import recognize_bound_form_structure

    payload = _payload(content)
    module = payload.get("module") if payload else None
    verification = (
        module.get("source_verification") if isinstance(module, dict) else None
    )
    citation = (
        verification.get("corpus_citation_path")
        if isinstance(verification, dict)
        else None
    )
    body = sources.get(citation) if isinstance(citation, str) else None
    if not isinstance(body, str):
        return False
    form = recognize_bound_form_structure(body, citation)
    return (
        form is not None
        and bind_ordered_form(form, content, sources, root_relative=root_relative)
        is not None
    )


def _wire_number(value: Any) -> Decimal | None:
    if not isinstance(value, dict) or set(value) != {"kind", "value"}:
        return None
    if value["kind"] == "integer" and type(value["value"]) is int:
        return Decimal(value["value"])
    if value["kind"] == "decimal" and isinstance(value["value"], str):
        try:
            parsed = Decimal(value["value"])
            return parsed if parsed.is_finite() else None
        except InvalidOperation:
            return None
    return None


def _compiled_expression(node: ast.expr, rules: Mapping[str, Any], inputs: set[str]):
    if isinstance(node, ast.Name):
        if node.id in inputs:
            return {"kind": "input", "name": node.id}
        if node.id in rules and rules[node.id]["kind"] == "derived":
            return {"kind": "derived", "name": node.id}
        if node.id in rules and rules[node.id]["kind"] == "parameter":
            return {
                "kind": "parameter_lookup",
                "parameter": node.id,
                "index": {"kind": "literal", "value": {"kind": "integer", "value": 0}},
            }
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return {
            "kind": "add",
            "items": [
                _compiled_expression(node.left, rules, inputs),
                _compiled_expression(node.right, rules, inputs),
            ],
        }
    if isinstance(node, ast.BinOp) and type(node.op) in (ast.Sub, ast.Mult):
        return {
            "kind": {ast.Add: "add", ast.Sub: "sub", ast.Mult: "mul"}[type(node.op)],
            "left": _compiled_expression(node.left, rules, inputs),
            "right": _compiled_expression(node.right, rules, inputs),
        }
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "min"
        and not node.keywords
        and len(node.args) == 2
    ):
        return {
            "kind": "min",
            "items": [_compiled_expression(n, rules, inputs) for n in node.args],
        }
    raise ValueError("Unsupported bound expression")


def observed_form_frame(
    binding: OrderedFormBinding, evidence: Any, content: str, cases: Any
) -> bool:
    """Observe this graph's declared frame, without changing Person evidence.

    Requires one complete, strictly explained case of this source-owned graph.
    Ordinary formula witnesses remain an independent caller prerequisite.
    """
    import json

    from .same_case_execution import case_key, strict_explain_metadata
    from .validator_pipeline import ValidatorPipeline

    try:
        if binding.consumer_namespace is None or not evidence.bound(content, ()):
            return False
        if (
            bind_ordered_form(
                binding.form,
                content,
                {binding.form.citation: binding.form.source},
                root_relative=evidence.root_relative,
            )
            != binding
        ):
            return False
        payload = _payload(content)
        rules = _named(payload, "rules")
        inputs = _named(payload, "inputs")
        derived = {n: r for n, r in rules.items() if r["kind"] == "derived"}
        parameters = {n: r for n, r in rules.items() if r["kind"] == "parameter"}
        documents = {d.rule_name: d for d in binding.document_requirements}
        row_derived = {n: r for n, r in derived.items() if n not in documents}
        row_inputs = {
            n: r
            for n, r in inputs.items()
            if n not in {d.input_name for d in documents.values()}
        }
        program = json.loads(evidence.artifact_bytes)["program"]
        declarations = {item["name"]: item for item in program["derived"]}
        constants = {item["name"]: item for item in program["parameters"]}
        if (
            program.get("relations") != []
            or set(declarations) != set(derived)
            or len(declarations) != len(program["derived"])
            or set(constants) != set(parameters)
            or len(constants) != len(program["parameters"])
        ):
            return False
        prefix = binding.consumer_namespace + "#"
        frame = {
            "effective_from": f"{binding.form.header.year}-01-01",
            "effective_to": f"{binding.form.header.year}-12-31",
        }
        expected_period = {
            "period_kind": "tax_year",
            "start": frame["effective_from"],
            "end": frame["effective_to"],
        }
        for name, rule in rules.items():
            declaration = (declarations if name in derived else constants)[name]
            if declaration["id"] != prefix + name:
                return False
            if name in documents:
                document = documents[name]
                expression = {
                    "kind": "comparison",
                    "left": {"kind": "input", "name": document.input_name},
                    "op": "eq",
                    "right": {
                        "kind": "literal",
                        "value": {"kind": "bool", "value": True},
                    },
                }
                if (
                    any(
                        declaration.get(k) != v
                        for k, v in {
                            "entity": document.entity,
                            "dtype": "judgment",
                            "unit": None,
                            "period": "Year",
                            "semantics": "judgment",
                            "source": rule.get("source"),
                            "corpus_citation_path": binding.form.citation,
                        }.items()
                    )
                    or json.dumps(declaration.get("expr"), sort_keys=True)
                    != json.dumps(expression, sort_keys=True)
                    or json.dumps(declaration["versions"], sort_keys=True)
                    != json.dumps(
                        [{**frame, "semantics": "judgment", "expr": expression}],
                        sort_keys=True,
                    )
                ):
                    return False
            elif name in derived:
                expression = _compiled_expression(
                    _tree(rule["versions"][0]["formula"]), rules, set(inputs)
                )
                if any(
                    declaration.get(k) != v
                    for k, v in {
                        "entity": binding.entity,
                        "dtype": "decimal",
                        "unit": binding.unit,
                        "period": "Year",
                        "semantics": "scalar",
                        "expr": expression,
                    }.items()
                ) or declaration["versions"] != [
                    {**frame, "semantics": "scalar", "expr": expression}
                ]:
                    return False
            else:
                versions = declaration["versions"]
                if (
                    declaration.get("unit") != rule.get("unit")
                    or len(versions) != 1
                    or set(versions[0]) != {"effective_from", "effective_to", "values"}
                    or any(versions[0][k] != v for k, v in frame.items())
                    or set(versions[0]["values"]) != {"0"}
                    or _wire_number(versions[0]["values"]["0"])
                    != _number(rule["versions"][0]["formula"])
                ):
                    return False
        if not observed_document_pairs(
            binding.document_requirements, evidence, cases, prefix, expected_period
        ):
            return False
        pipeline = object.__new__(ValidatorPipeline)
        for case in cases or ():
            if not isinstance(case, dict) or case.get("period") != expected_period:
                continue
            records = [r for r in evidence.records if r.case_sha256 == case_key(case)]
            if len(records) != 1:
                continue
            from types import SimpleNamespace

            rows = {row.number: row for row in binding.form.rows}
            if any(
                not binding.matches(
                    content=content,
                    branch=SimpleNamespace(
                        start=rows[n].span[0],
                        end=rows[n].span[1],
                        text=binding.form.source[slice(*rows[n].span)],
                    ),
                    rule=derived[name],
                    principal_rules=derived,
                    case=case,
                    leaf=str(derived[name]["versions"][0]["formula"]),
                    trace=(),
                )
                for n, name in binding.row_owners
                if name in derived
            ):
                continue
            record = records[0]
            request = json.loads(record.request_json)
            response = json.loads(record.response_json)
            if (
                not strict_explain_metadata(request, response)
                or len(request["queries"]) != 1
                or len(response["results"]) != 1
            ):
                continue
            query = request["queries"][0]
            result = response["results"][0]
            entity_id = pipeline._rulespec_case_query_entity_id(
                case, binding.entity, record.case_index
            )
            if (
                query.get("entity_id") != entity_id
                or query.get("period") != expected_period
                or query.get("assessment_date") is not None
                or any(
                    result.get(k) != query.get(k)
                    for k in ("entity_id", "period", "assessment_date")
                )
            ):
                continue
            outputs = {prefix + n for n in row_derived}
            if (
                set(query["outputs"]) != outputs
                or len(query["outputs"]) != len(outputs)
                or set(result["outputs"]) != outputs
                or set(result["trace"]) != outputs
                or set(case["output"]) != outputs
            ):
                continue
            facts = request["dataset"]["inputs"]
            if (
                len(facts) != len(row_inputs)
                or request["dataset"].get("relations") != []
                or {f["name"] for f in facts}
                != {prefix + "input." + n for n in row_inputs}
            ):
                continue
            case_inputs = {
                k.rsplit("#", 1)[-1].removeprefix("input."): v
                for k, v in case["input"].items()
            }
            if any(
                f.get("entity") != binding.entity
                or f.get("entity_id") != entity_id
                or f.get("interval")
                != {"start": frame["effective_from"], "end": frame["effective_to"]}
                or _wire_number(f.get("value"))
                != _number(case_inputs.get(f["name"].split("#input.", 1)[1]))
                for f in facts
            ):
                continue
            valid = True
            for name, rule in row_derived.items():
                ref = prefix + name
                node = result["trace"][ref]
                output = result["outputs"][ref]
                for observed in (node, output):
                    if (
                        any(
                            observed.get(k) != v
                            for k, v in {
                                "id": ref,
                                "name": name,
                                "kind": "scalar",
                                "dtype": "decimal",
                                "unit": binding.unit,
                            }.items()
                        )
                        or _wire_number(observed.get("value")) is None
                        or _wire_number(observed.get("value"))
                        != _number(case["output"][ref])
                    ):
                        valid = False
                if node.get("entity_id") != entity_id:
                    valid = False
                expression = node.get("executed_expression")
                if not isinstance(expression, str):
                    valid = False
                    break
                for local in sorted(rules, key=len, reverse=True):
                    suffix = "[0]" if local in parameters else ""
                    expression = expression.replace(prefix + local + suffix, local)
                if name in dict(binding.entered):
                    expected_input = dict(binding.entered)[name]
                    expression = expression.replace(
                        'input("' + expected_input + '")', expected_input
                    )
                parsed = _tree(expression)
                if parsed is None or ast.dump(parsed) != dict(binding.formulas)[name]:
                    valid = False
                expected_dependencies = {
                    prefix + n.id
                    for n in ast.walk(_tree(rule["versions"][0]["formula"]))
                    if isinstance(n, ast.Name) and n.id in derived
                }
                if (
                    not isinstance(node.get("dependencies"), list)
                    or set(node["dependencies"]) != expected_dependencies
                    or len(node["dependencies"]) != len(expected_dependencies)
                ):
                    valid = False
            if valid:
                return True
        return False
    except (
        ValueError,
        TypeError,
        KeyError,
        AttributeError,
        IndexError,
        RecursionError,
    ):
        return False
