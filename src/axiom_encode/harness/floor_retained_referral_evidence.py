"""Local signed retention and authenticated informational referrals.

Nothing here evaluates a referred policy or decides loss eligibility. Evidence is
allocated only to the complete, closed printed row after both local values run.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
from dataclasses import dataclass
from decimal import Decimal
from itertools import combinations
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from axiom_encode.corpus_resolver import (
    CorpusResolutionError,
    read_bounded_regular_file,
    resolve_local_corpus_source,
)

from .actual_floor_evidence import (
    _changed_primitive,
    _explicit_identity,
    _query,
    _subtraction,
)
from .imported_export_evidence import ResolvedImportedExport, _atoms, _named, _payload
from .printed_signed_format_evidence import _closed_frame, _exact_source
from .printed_signed_format_evidence import _subtraction as _signed
from .same_case_execution import SameCaseEvidence, case_key, formula_witnesses

_SEAL = object()
_ROW = re.compile(
    r"Line (?P<row>[1-9]\d{0,2}) and line (?P<field>[1-9]\d{4}): "
    r"(?P<operation>Line (?P<left>[1-9]\d{0,2}) minus line (?P<right>[1-9]\d{0,2})) "
    r'\(if negative, enter "0"\) If negative, you may have a non-capital loss '
    r"\(see Form (?P<form>[A-Z][A-Z0-9]*)\) and the negative amount is to be used "
    r"for certain calculations \(go to (?P<url>[a-z0-9.-]+/line-(?P<url_field>[1-9]\d{4}))\) "
    r"(?P<caption>[A-Za-z][A-Za-z ]*) = \^"
)
_CONDITION = re.compile(
    r"\b(if|unless|when|only|eligible|eligibility|must|may|shall|except|required|provided|subject)\b",
    re.I,
)


def _sha(value: str | bytes) -> str:
    return hashlib.sha256(
        value.encode() if isinstance(value, str) else value
    ).hexdigest()


@dataclass(frozen=True)
class ReferralSource:
    citation: str
    body: str
    attestation_json: str
    year: str


@dataclass(frozen=True)
class BoundReferralContext:
    consumer_sha256: str
    consumer_file: Path
    consumer_root: Path
    target_attestation_json: str
    target: str
    target_file: Path
    target_content: str
    target_family: str
    relation_name: str
    sources: tuple[ReferralSource, ...]
    seal: object

    def bound(self, content: str, context: Mapping[str, str | None]) -> bool:
        return (
            self.seal is _SEAL
            and self.consumer_sha256 == _sha(content)
            and _read(self.consumer_root, self.consumer_file) == content
            and _read(self.consumer_root, self.target_file) == self.target_content
            and len({s.citation for s in self.sources}) == len(self.sources)
            and all(context.get(s.citation) == s.body for s in self.sources)
        )


def _read(root: Path, path: Path) -> str:
    return read_bounded_regular_file(
        root, path, label="referral source module", max_bytes=10 * 1024 * 1024
    ).decode("utf-8")


def collect_referral_context(
    pipeline: Any,
    content: str,
    rules_file: Path | None,
    proof_sources: Mapping[str, str | None] | None,
) -> BoundReferralContext | None:
    """Resolve only an explicitly cited target, under the existing release/root."""
    from . import validator_pipeline as v

    try:
        payload = _payload(content)
        rules = _named(payload, "rules") if payload else None
        if rules_file is None or rules is None or not proof_sources:
            return None
        candidates = [
            r
            for r in rules.values()
            if r.get("kind") == "source_relation"
            and isinstance(r.get("source_relation"), dict)
            and r["source_relation"].get("type") == "cites"
        ]
        if len(candidates) != 1:
            return None
        relation = candidates[0]
        if any(
            k in relation
            for k in (
                "versions",
                "formula",
                "entity",
                "period",
                "dtype",
                "unit",
                "default",
                "arity",
                "indexed_by",
            )
        ):
            return None
        edge = relation["source_relation"]
        if set(edge) != {"type", "target"} or not isinstance(edge["target"], str):
            return None
        ref = v._parse_rulespec_target(edge["target"])
        if ref is None or ref.symbol is not None:
            return None
        root = pipeline._validation_source_root(rules_file)
        target = v._resolve_rulespec_target_file(
            ref, root, rulespec_dependency_roots=pipeline.rulespec_dependency_roots
        )
        if target is None:
            return None
        if _read(root, rules_file) != content:
            return None
        target_content = _read(root, target)
        provider = _payload(target_content)
        module = provider.get("module") if provider else None
        verification = (
            module.get("source_verification") if isinstance(module, dict) else None
        )
        family = (
            verification.get("corpus_citation_path")
            if isinstance(verification, dict)
            else None
        )
        if (
            not isinstance(family, str)
            or edge["target"].replace(":policies/", "/policy/") != family
        ):
            return None
        release = v._AUTHORITATIVE_CORPUS_RELEASE.get()
        if release is None:
            return None
        target_source = resolve_local_corpus_source(family, release)
        target_attestation = target_source.to_attestation()
        if "corpus_citation_paths" in verification:
            return None
        # Legacy informational targets may omit pins, but any supplied pin must
        # agree with the independently resolved current authenticated release.
        for key in ("source_sha256", "corpus_release", "corpus_release_content_sha256"):
            if key in verification and verification[key] != target_attestation.get(key):
                return None
        sources = []
        # These are the normal consumer's direct proof bodies, never recursive
        # target imports or a replacement numeric/source context.
        for citation, body in proof_sources.items():
            if not isinstance(citation, str) or not isinstance(body, str):
                return None
            source = resolve_local_corpus_source(citation, release)
            if source.proof_evidence_text != body or source.body != body:
                return None
            date = source.row.expression_date
            if not isinstance(date, str) or not re.fullmatch(
                r"\d{4}-\d{2}-\d{2}", date
            ):
                return None
            sources.append(
                ReferralSource(
                    citation,
                    body,
                    json.dumps(source.to_attestation(), sort_keys=True),
                    date[:4],
                )
            )
        return BoundReferralContext(
            _sha(content),
            rules_file,
            root,
            json.dumps(target_attestation, sort_keys=True),
            edge["target"],
            target,
            target_content,
            family,
            relation["name"],
            tuple(sources),
            _SEAL,
        )
    except (
        OSError,
        ValueError,
        TypeError,
        AttributeError,
        yaml.YAMLError,
        CorpusResolutionError,
    ):
        return None


def _referrals(
    binding: BoundReferralContext,
    payload: dict,
    match: re.Match,
    citation: str,
    source_text: str,
    context: Mapping[str, str | None],
    year: str,
    signed_rule: dict,
) -> bool:
    rules = _named(payload, "rules")
    if rules is None or binding.relation_name not in rules:
        return False
    relation = rules[binding.relation_name]
    if relation.get("source_relation") != {"type": "cites", "target": binding.target}:
        return False
    local = _atoms(relation, "source_relation.type", "exception")
    targets = _atoms(relation, "source_relation.target", "definition")
    if len(local) != 1 or len(targets) != 2:
        return False
    source = _exact_source(local[0], context)
    expected = (
        f"If negative, you may have a non-capital loss (see Form {match['form']})"
    )
    if (
        source is None
        or source[0] != citation
        or source[1].strip() != source_text
        or source[2] != expected
    ):
        return False
    target_sources = [_exact_source(a, context) for a in targets]
    if any(
        s is None or s[0].rsplit("/", 1)[0] != binding.target_family
        for s in target_sources
    ):
        return False
    if not any(
        re.fullmatch(
            re.escape(match["form"]) + r" E \(" + re.escape(year[-2:]) + r"\)", s[2]
        )
        for s in target_sources
    ):
        return False
    if not any(
        re.fullmatch(
            r"Use this form to request a carryback of a loss from "
            + re.escape(year)
            + r" to any of the three previous tax years\.",
            s[2],
        )
        for s in target_sources
    ):
        return False
    years = {s.citation: s.year for s in binding.sources}
    if any(years.get(s[0]) != year for s in target_sources):
        return False
    # Definition evidence corroborates retained-value meaning, not an outgoing
    # citation to either downstream calculation or its existing RuleSpec owner.
    definitions = _atoms(signed_rule, "source", "definition")
    if len(definitions) != 2 or not isinstance(signed_rule.get("source"), str):
        return False
    families = set()
    for atom in definitions:
        source = _exact_source(atom, context)
        if (
            source is None
            or source[0].split("/")[0] != citation.split("/")[0]
            or years.get(source[0]) != year
        ):
            return False
        text = " ".join(source[2].split())
        pattern = (
            r"(?:Line [1-9]\d*: Net |)income(?: amount)? from line "
            + re.escape(match["field"])
            + r" of the return \(or the amount you would have entered if the instructions on line "
            + re.escape(match["field"])
            + r' said "if negative, show in brackets"\)'
        )
        if re.fullmatch(pattern, text) is None:
            return False
        family = source[0].rsplit("/", 1)[0]
        if family in (citation.rsplit("/", 1)[0], binding.target_family):
            return False
        families.add(family)
    return len(families) == 2


def witnesses(
    evidence: SameCaseEvidence,
    branch: Any,
    *,
    content: str,
    source_text: str,
    corpus_citation_path: str,
    principal_rules: dict,
    asserted_by_rule: dict,
    formula_environment: dict,
    referral_context: BoundReferralContext | None = None,
    source_context: Mapping[str, str | None] | None = None,
    imported_exports: Sequence[ResolvedImportedExport] = (),
    printed_operations: Any = None,
    **ordinary_context: Any,
) -> set[tuple]:
    """Optional evidence must refuse malformed data without changing diagnostics."""
    try:
        return _witnesses(
            evidence,
            branch,
            content=content,
            source_text=source_text,
            corpus_citation_path=corpus_citation_path,
            principal_rules=principal_rules,
            asserted_by_rule=asserted_by_rule,
            formula_environment=formula_environment,
            referral_context=referral_context,
            source_context=source_context,
            imported_exports=imported_exports,
            printed_operations=printed_operations,
            **ordinary_context,
        )
    except (
        OSError,
        ValueError,
        TypeError,
        KeyError,
        AttributeError,
        IndexError,
        RecursionError,
        yaml.YAMLError,
    ):
        return set()


def _witnesses(
    evidence,
    branch,
    *,
    content,
    source_text,
    corpus_citation_path,
    principal_rules,
    asserted_by_rule,
    formula_environment,
    referral_context,
    source_context,
    imported_exports,
    printed_operations,
    **ordinary_context,
):
    from . import source_completeness as s

    if (
        not isinstance(referral_context, BoundReferralContext)
        or not source_context
        or not referral_context.bound(content, source_context)
        or not evidence.bound(content, imported_exports)
        or printed_operations is None
        or source_text[branch.start : branch.end] != branch.text
        or source_text.count(branch.text) != 1
    ):
        return set()
    match = _ROW.fullmatch(branch.text)
    if (
        match is None
        or match["field"] != match["url_field"]
        or _CONDITION.search(match["caption"])
    ):
        return set()
    # Projection is an existing source/grid/conflict-bound capability, not a
    # free mask. Requiring it also proves full physical row ownership.
    if (
        printed_operations.operation_projection(
            branch, source_text=source_text, citation_path=corpus_citation_path
        )
        is None
    ):
        return set()
    payload = _payload(content)
    rules = _named(payload, "rules") if payload else None
    if rules is None or any(rules.get(n) != r for n, r in principal_rules.items()):
        return set()
    context = dict(source_context)
    owners = []
    for name, rule in principal_rules.items():
        versions = rule.get("versions")
        if (
            not isinstance(versions, list)
            or not versions
            or not all(isinstance(v, dict) for v in versions)
        ):
            continue
        shapes = [_subtraction(v.get("formula", "")) for v in versions]
        if any(x is None for x in shapes) or len({ast.dump(x) for x in shapes}) != 1:
            continue
        if branch.required_principal_output not in (None, name):
            continue
        owners.append((name, rule, shapes[0]))
    if len(owners) != 1:
        return set()
    owner, floor, subtraction = owners[0]
    frames = [
        _closed_frame(floor, i, v, context) for i, v in enumerate(floor["versions"])
    ]
    if not frames or None in frames or len(set(frames)) != len(frames):
        return set()
    signed = []
    for name, rule in principal_rules.items():
        versions = rule.get("versions")
        if not isinstance(versions, list) or len(versions) != len(frames):
            continue
        if any(
            rule.get(k) != floor.get(k)
            for k in ("kind", "dtype", "entity", "period", "unit")
        ):
            continue
        if any(
            not isinstance(v, dict)
            or _signed(v.get("formula")) is None
            or ast.dump(_signed(v["formula"])) != ast.dump(subtraction)
            for v in versions
        ):
            continue
        for i, version in enumerate(versions):
            atoms = _atoms(rule, f"versions[{i}].formula", "formula")
            proof = _exact_source(atoms[0], context) if len(atoms) == 1 else None
            expected = branch.text[: branch.text.index(" (if negative")]
            if (
                proof is None
                or proof[0] != corpus_citation_path
                or proof[1].strip() != source_text
                or proof[2] != expected
                or _closed_frame(rule, i, version, context) != frames[i]
                or not _referrals(
                    referral_context,
                    payload,
                    match,
                    corpus_citation_path,
                    source_text,
                    context,
                    str(version.get("effective_from", ""))[:4],
                    rule,
                )
            ):
                break
        else:
            signed.append((name, rule))
    if len(signed) != 1:
        return set()
    raw_name, raw_rule = signed[0]
    artifact = json.loads(evidence.artifact_bytes)
    declarations = artifact["program"]["derived"]
    module = evidence.root_relative.removesuffix(".yaml").replace("/", ":", 1)
    for name in (owner, raw_name, subtraction.left.id, subtraction.right.id):
        found = [d for d in declarations if d.get("id") == module + "#" + name]
        if len(found) != 1 or any(
            found[0].get(k) != value
            for k, value in {
                "entity": "Person",
                "period": "Year",
                "dtype": "decimal",
                "unit": floor.get("unit"),
                "semantics": "scalar",
            }.items()
        ):
            return set()
    observations = []
    for case in asserted_by_rule.get(owner, ()):
        if not isinstance(case.get("output"), dict) or case not in asserted_by_rule.get(
            raw_name, ()
        ):
            continue
        query = _query(evidence, case)
        values = evidence.values(case, content)
        if query is None or values is None:
            continue
        imported, observed = values
        if imported.keys() & formula_environment.keys():
            continue
        environment = {**formula_environment, **imported}
        dependencies = s._case_dependency_environment(
            principal_rules,
            case,
            formula_environment=environment,
            require_asserted_value=False,
        )
        needed = (owner, raw_name, subtraction.left.id, subtraction.right.id)
        if any(n not in dependencies or n not in observed for n in needed) or any(
            n not in observed
            or not s._formula_runtime_values_equal(value, observed[n])
            or not s._asserted_formula_runtime_values_equal(
                principal_rules[n], value, s._test_case_asserted_output_value(case, n)
            )
            for n, value in dependencies.items()
        ):
            continue
        actual = {**environment, **dependencies}
        execution = s._case_formula_execution(floor, case, formula_environment=actual)
        raw_execution = s._case_formula_execution(
            raw_rule, case, formula_environment=actual
        )
        if (
            execution is None
            or raw_execution is None
            or raw_execution.trace
            or _signed(raw_execution.leaf) is None
            or ast.dump(_signed(raw_execution.leaf)) != ast.dump(subtraction)
            or not printed_operations.local_operands_bind(
                branch,
                rule=floor,
                principal_rules=principal_rules,
                case=case,
                execution=execution,
            )
        ):
            continue
        a, b = (actual[n] for n in (subtraction.left.id, subtraction.right.id))
        if any(
            isinstance(v, bool) or not isinstance(v, (int, Decimal)) for v in (a, b)
        ):
            continue
        value = Decimal(a) - Decimal(b)
        if observed[raw_name] != value or observed[owner] != max(Decimal(0), value):
            continue
        if value > 0 and not (
            s._formula_branch_test_witnesses(
                branch,
                source_text=source_text,
                corpus_citation_path=corpus_citation_path,
                principal_rules=principal_rules,
                rule_names={owner},
                asserted_by_rule={owner: [case]},
                formula_environment=environment,
                dependency_cache={id(case): dependencies},
                execution_cache={(owner, id(case)): execution},
                printed_operations=printed_operations,
                **ordinary_context,
            )
            or formula_witnesses(
                evidence,
                branch,
                content=content,
                source_text=source_text,
                corpus_citation_path=corpus_citation_path,
                principal_rules=principal_rules,
                rule_names={owner},
                asserted_by_rule={owner: [case]},
                formula_environment=formula_environment,
                printed_operations=printed_operations,
                **ordinary_context,
            )
        ):
            continue
        selected = s._selected_rule_formula_version_index(floor, case)
        observations.append((case, query, value, selected))
    pairs = {i: set() for i in range(len(frames))}
    for (left, lq, lv, li), (right, rq, rv, ri) in combinations(observations, 2):
        if li != ri or li not in pairs or not (lv < 0 < rv or rv < 0 < lv):
            continue
        if (
            _explicit_identity(left) != _explicit_identity(right)
            or any(lq.get(k) != rq.get(k) for k in ("period", "assessment_date"))
            or left["output"].keys() != right["output"].keys()
            or _changed_primitive(evidence, left, right) != floor.get("unit")
        ):
            continue
        pairs[li].add(tuple(sorted((case_key(left), case_key(right)))))
    if not pairs or any(not values for values in pairs.values()):
        return set()
    return {
        (
            "floor-retained-referral",
            owner,
            raw_name,
            _sha(content),
            _sha(source_text),
            (branch.start, branch.end),
            referral_context.target,
            tuple((i, tuple(sorted(values))) for i, values in sorted(pairs.items())),
        )
    }
