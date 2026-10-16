"""Bind the closed worksheet operation block to selected proofs and executions."""

import ast
import dataclasses
import hashlib

from axiom_encode.harness import source_completeness as sc
from axiom_encode.harness import source_path_evidence as pe
from axiom_encode.harness import worksheet_operation_evidence as op
from axiom_encode.harness import worksheet_replacement_evidence as replacement

SPAN = (1119, 1492)
YEAR = ("2025-01-01", "2025-12-31")
AMT = frozenset(("bc_amt", "ontario_small", "ontario_large", "other_amt"))


def identity(value):
    return replacement.identity({"value": value})


@dataclasses.dataclass(frozen=True)
class Composition:
    resource: tuple = ()
    execution_records: tuple = ()
    ordinary_paths: tuple = ()
    claim_inventory: tuple = ()
    unresolved: tuple = ()
    input_identity: str = ""
    context_identity: str = ""
    source_identity: str = ""
    projection: tuple = ()
    selected_identity: tuple = ()

    def matches(self, bound, branch, principal_rules):
        try:
            rules = op.paths._index(bound.payload.get("rules")) or {}
            return (
                bool(self.resource)
                and not self.unresolved
                and (
                    (branch.start, branch.end) == SPAN
                    and branch.text == bound.source[slice(*SPAN)]
                    and type(branch.start) is int
                    and type(branch.end) is int
                    and self.input_identity
                    == pe.path_input_identity(bound.payload, bound.cases)
                    and self.context_identity == identity(bound.context)
                    and self.source_identity
                    == hashlib.sha256(bound.source.encode()).hexdigest()
                    and self.projection == bound.bundle.flat_case_indices
                    and bound.bundle.matches(
                        bound.payload, bound.cases, bound.source, bound.context
                    )
                    and bound.bundle.scalar_operations.matches(
                        bound.payload,
                        [bound.cases[i] for i in self.projection],
                        bound.source,
                        bound.context,
                    )
                    and all(
                        n in rules
                        and r == rules[n]
                        and getattr(r, "resolved_currency_rounding", None)
                        == getattr(rules[n], "resolved_currency_rounding", None)
                        for n, r in principal_rules.items()
                    )
                    and set(op._OUTPUTS) <= principal_rules.keys()
                    and self.selected_identity
                    == tuple(
                        (
                            name,
                            identity(rules[name]),
                            identity(
                                getattr(rules[name], "resolved_currency_rounding", None)
                            ),
                        )
                        for name, _, _ in self.selected_identity
                    )
                )
            )
        except (TypeError, ValueError, KeyError, IndexError, AttributeError):
            return False


def certify(bound):
    """Compose source operations, selected proofs and independent executions."""

    def fail(s):
        return Composition(unresolved=(s,))

    try:
        p, cases, source, context, bundle = (
            bound.payload,
            bound.cases,
            bound.source,
            bound.context,
            bound.bundle,
        )
        if not bundle.matches(p, cases, source, context):
            return fail("full identity")
        if hashlib.sha256(source.encode()).hexdigest() != replacement.PRIMARY_SHA:
            return fail("source identity")
        indices = bundle.flat_case_indices
        flat = [cases[i] for i in indices]
        core = bundle.scalar_operations
        if (
            not core.matches(p, flat, source, context)
            or core.unresolved
            or set(core.outputs) != set(op._OUTPUTS)
        ):
            return fail("universal core")
        if (
            bundle.scalar_paths.input_identity != pe.path_input_identity(p, flat)
            or bundle.scalar_paths.unresolved
        ):
            return fail("path identity")
        expected_indices = tuple(
            i
            for i, c in enumerate(cases)
            if not any(
                sc._test_case_asserted_output_value(c, n)
                is not sc._UNRESOLVED_CONDITION_VALUE
                for n in replacement.tr.OUTPUTS
            )
        )
        if indices != expected_indices or len(set(indices)) != len(indices):
            return fail("projection")
        quotes = sc._annual_source_quoted_spans(source)
        if quotes is None or any(a < SPAN[1] and b > SPAN[0] for a, b in quotes):
            return fail("quoted span")
        rules = op.paths._index(p.get("rules")) or {}
        contract = op.paths.worksheet_shadow_contract(
            source,
            citation=op._PRIMARY,
            line2_owner=op._LINE2,
            tax_otherwise_owner=op._TAX,
        )
        if contract.unresolved:
            return fail("source contract")
        ordinary = [o for o in contract.obligations if o.identity == "ordinary_line2"]
        if len(ordinary) != 1:
            return fail("ordinary descriptor")
        ordinary = ordinary[0]
        inspected = op.paths.inspect_source_paths(
            p, source_text=source, owner=op._LINE2, obligations=contract.obligations
        )
        rows = [
            r
            for r in inspected.rows
            if r["obligation"] == ordinary.identity and r["status"] != "excluded"
        ]
        if (
            inspected.unresolved
            or not rows
            or any(r["status"] != "diagnostic_match" for r in rows)
        ):
            return fail("ordinary compatible paths")
        ordinary_paths = tuple(sorted(r["path"] for r in rows))
        amt_records = [
            r for r in bundle.scalar_paths.obligations if r.owner == op._LINE2
        ]
        if (
            len(amt_records) != 4
            or {r.identity for r in amt_records} != AMT
            or any(
                not r.certified
                or r.interval != YEAR
                or r.citation != op._PRIMARY
                or r.source_sha256 != replacement.PRIMARY_SHA
                for r in amt_records
            )
        ):
            return fail("four AMT records")
        for record in amt_records:
            obligations = [
                o
                for o in contract.obligations
                if o.identity == record.identity and o.owner == op._LINE2
            ]
            current_rows = [
                row
                for row in inspected.rows
                if row["obligation"] == record.identity and row["status"] != "excluded"
            ]
            if (
                len(obligations) != 1
                or record.source_spans != obligations[0].spans
                or not current_rows
                or any(row["status"] != "diagnostic_match" for row in current_rows)
                or record.compatible_paths != tuple(row["path"] for row in current_rows)
            ):
                return fail("AMT source/path record differs")
            for dependency in record.selected_dependencies:
                rule = rules.get(dependency.name)
                if rule is None or not any(
                    i == dependency.version
                    and (lo, hi) == (dependency.start, dependency.end)
                    and version == dependency.formula
                    for i, version, lo, hi in sc._effective_formula_version_intervals(
                        rule
                    )
                ):
                    return fail("AMT selected dependency differs")
        # Resolve all source-defined selected computations independently. LINE2 is
        # atomic in the operation resolver, so its source paths are checked above.
        selected = {}
        for owner, expected in op._expected_operations().items():
            resolver = op._OperationResolver(p, owner, *YEAR, source, op._PRIMARY)
            actual = resolver.expand(ast.Name(id=owner))
            if (
                actual is None
                or resolver.errors
                or not op._matches_source_operation(owner, actual, expected)
            ):
                return fail("source operation:" + owner)
            selected.update(resolver.dependencies)
        for path in inspected.paths:
            selected.update({d.name: d for d in path.dependencies})
        for owner in op._OUTPUTS:
            rule = rules[owner]
            d = selected.get(owner)
            if (
                d is None
                or (d.start, d.end) != YEAR
                or (
                    rule.get("dtype"),
                    rule.get("unit"),
                    rule.get("entity"),
                    rule.get("period"),
                )
                != ("Money", "CAD", "Person", "Year")
            ):
                return fail("selected owner")
        line2 = rules[op._LINE2]
        version = selected[op._LINE2].version
        if (
            len(ordinary.spans) != 1
            or (ordinary.spans[0].start, ordinary.spans[0].end) != (1223, 1309)
            or source[1305:1309] != " – 2"
            or (1308, 1309) not in sc._corroborated_form_output_label_spans(source)
        ):
            return fail("ordinary layout boundary")
        if not any(
            path == f"versions[{version}].formula"
            and cite == op._PRIMARY
            and source[1223:1305] in excerpt
            for path, cite, excerpt in sc._rule_source_excerpt_atoms(line2)
        ):
            return fail("ordinary owned proof")
        # Exact selected worksheet-owner claims only: a partial donor, foreign
        # citation, or unselected version cannot borrow the universal core.
        claims = []
        for name, rule in rules.items():
            for path, cite, excerpt in sc._rule_source_excerpt_atoms(rule):
                ver = sc._formula_proof_version_index(path)
                if ver is None or not excerpt:
                    continue
                if source.count(excerpt) == 1:
                    a = source.index(excerpt)
                    b = a + len(excerpt)
                    overlap = a < SPAN[1] and SPAN[0] < b
                else:
                    x, y = (
                        sc._collapse_text(excerpt),
                        sc._collapse_text(source[slice(*SPAN)]),
                    )
                    overlap = bool(x) and (x in y or y in x)
                    a, b = -1, -1
                if not overlap:
                    continue
                if (
                    name not in op._OUTPUTS
                    or name not in selected
                    or ver != selected[name].version
                    or cite != op._PRIMARY
                    or (selected[name].start, selected[name].end) != YEAR
                    or a < 0
                ):
                    return fail("unselected or ambiguous worksheet claimant")
                claims.append((name, ver, a, b, identity(rule)))
        constants = sc._constant_rule_environment(p)
        executions = {}
        records = []
        for owner, fi in core.case_links:
            if (
                owner not in op._OUTPUTS
                or type(fi) is not int
                or not 0 <= fi < len(flat)
            ):
                return fail("case link")
            case = flat[fi]
            rule = rules[owner]
            if (
                sc._selected_rule_formula_version_index(rule, case)
                != selected[owner].version
            ):
                return fail("case selected version")
            deps = sc._case_dependency_environment(
                rules, case, formula_environment=constants, require_asserted_value=False
            )
            execution = sc._case_formula_execution(
                rule, case, formula_environment=constants, dependency_environment=deps
            )
            expected = sc._test_case_asserted_output_value(case, owner)
            if (
                execution is None
                or expected is sc._UNRESOLVED_CONDITION_VALUE
                or not sc._asserted_formula_runtime_values_equal(
                    rule, sc._formula_execution_runtime_value(execution), expected
                )
            ):
                return fail("independent execution:" + owner + ":" + str(fi))
            executions[owner, fi] = execution
            records.append(
                (
                    owner,
                    indices[fi],
                    selected[owner].version,
                    identity(case),
                    identity(rule),
                    identity(dataclasses.asdict(execution)),
                )
            )
        if {r[0] for r in records} != set(op._OUTPUTS):
            return fail("empty output coverage")
        # Prove ordinary false-AMT execution and every AMT compatible path using
        # independently recomputed results, never asserted dependency values.
        for path_index in ordinary_paths:
            path = inspected.paths[path_index]
            found = False
            for fi, case in enumerate(flat):
                env = sc._case_input_formula_environment(case) or {}
                execution = executions.get((op._LINE2, fi))
                if execution is None or env.get("minimum_tax_is_payable") is not False:
                    continue
                if not all(
                    op.paths._predicate_value(pred, env) is True
                    for pred in path.predicates
                ):
                    continue
                if sc._formula_execution_runtime_value(execution) != env.get(
                    op._CREDIT
                ):
                    continue
                # Ordinary path must actually test AMT and take its false arm.
                if not any(
                    step.kind == "if"
                    and step.selectors == ("minimum_tax_is_payable",)
                    and step.choice == 1
                    or step.kind == "if"
                    and step.selectors == ("not minimum_tax_is_payable",)
                    and step.choice == 0
                    for step in execution.trace
                ):
                    continue
                found = True
            if not found:
                return fail("unexecuted ordinary path")
        for record in amt_records:
            for path_index in record.compatible_paths:
                if not 0 <= path_index < len(inspected.paths):
                    return fail("AMT path index")
                path = inspected.paths[path_index]
                if not any(
                    (op._LINE2, fi) in executions
                    and (sc._case_input_formula_environment(flat[fi]) or {}).get(
                        "minimum_tax_is_payable"
                    )
                    is True
                    and all(
                        op.paths._predicate_value(
                            pred, sc._case_input_formula_environment(flat[fi]) or {}
                        )
                        is True
                        for pred in path.predicates
                    )
                    for fi in record.case_links
                ):
                    return fail("unexecuted AMT path")
        immutable = (
            replacement.PRIMARY_SHA,
            SPAN,
            tuple(sorted(claims)),
            tuple(sorted(records)),
            ordinary_paths,
            identity(amt_records),
        )
        return Composition(
            ("worksheet-composition", identity(immutable)),
            tuple(sorted(records)),
            ordinary_paths,
            tuple(sorted(claims)),
            (),
            pe.path_input_identity(p, cases),
            identity(context),
            replacement.PRIMARY_SHA,
            indices,
            tuple(
                (
                    name,
                    identity(rules[name]),
                    identity(getattr(rules[name], "resolved_currency_rounding", None)),
                )
                for name in sorted(selected)
            ),
        )
    except (
        TypeError,
        ValueError,
        KeyError,
        IndexError,
        AttributeError,
        OverflowError,
    ) as error:
        return fail("malformed:" + type(error).__name__ + ":" + str(error))


@dataclasses.dataclass(frozen=True)
class BoundWorksheetComposition:
    """One closed operation resource bound to current analyzer inputs."""

    evidence: Composition
    bound: replacement.BoundWorksheetReplacement

    def resource_for(self, branch, principal_rules):
        return (
            self.evidence.resource
            if self.evidence.matches(self.bound, branch, principal_rules)
            else None
        )


def bind_worksheet_composition(bound):
    return BoundWorksheetComposition(certify(bound), bound)
