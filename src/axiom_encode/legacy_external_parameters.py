"""Conservative replacement obligations for externally proved scalar parameters.

This is input admission, not a source validator or a generated-output exemption.
Every admitted rule must survive unchanged and pass the ordinary proof pipeline.
"""

import re
from decimal import Decimal, InvalidOperation


def external_parameter_obligations(
    payload: dict, citation: str, segments
) -> list[dict]:
    rules = payload.get("rules")
    if not isinstance(rules, list):
        raise ValueError("external source admission requires a rules list")
    names = [rule.get("name") for rule in rules if isinstance(rule, dict)]
    if (
        len(names) != len(rules)
        or any(not isinstance(n, str) or not n for n in names)
        or len(set(names)) != len(names)
    ):
        raise ValueError("external source admission requires unique named rules")
    required = []
    admitted_reference_count = 0
    for rule in rules:
        metadata = rule.get("metadata") or {}
        if not isinstance(metadata, dict):
            raise ValueError("external source admission requires structured metadata")
        proof = metadata.get("proof") or {}
        if not isinstance(proof, dict):
            raise ValueError("external source admission requires structured proof")
        atoms = proof.get("atoms") or []
        if not isinstance(atoms, list):
            raise ValueError("external source admission requires proof atoms")
        cited = [
            atom
            for atom in atoms
            if isinstance(atom, dict)
            and isinstance(atom.get("source"), dict)
            and atom["source"].get("corpus_citation_path") == citation
        ]
        if not cited:
            continue
        versions = rule.get("versions")
        if (
            rule.get("kind") != "parameter"
            or rule.get("indexed_by") is not None
            or not isinstance(versions, list)
            or len(versions) != 1
            or not isinstance(versions[0], dict)
        ):
            raise ValueError(
                f"external source {citation} requires an unchanged scalar parameter"
            )
        value = versions[0].get("formula")
        if (
            isinstance(value, bool)
            or not isinstance(value, (str, int, float))
            or not re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)", str(value).strip())
        ):
            raise ValueError(
                f"external source {citation} requires a literal numeric parameter"
            )
        try:
            if not Decimal(str(value).strip()).is_finite():
                raise ValueError("nonfinite scalar parameter")
        except InvalidOperation as exc:
            raise ValueError("invalid scalar parameter") from exc
        for atom in cited:
            excerpt = atom["source"].get("excerpt")
            if (
                atom.get("path") != "versions[0].formula"
                or not isinstance(excerpt, str)
                or not excerpt.strip()
                or not any(
                    " ".join(excerpt.split()) in " ".join(segment.split())
                    for segment in segments
                )
            ):
                raise ValueError(
                    f"external source {citation} lacks a source-grounded scalar formula proof"
                )
        admitted_reference_count += len(cited)
        required.append(rule)
    if not required:
        raise ValueError(f"external source {citation} has no scalar proof obligation")

    reference_counts = {}

    def count_references(node, ancestors=frozenset()):
        if not isinstance(node, (dict, list)):
            return 0
        if id(node) in ancestors:
            raise ValueError("external source admission rejects cyclic YAML")
        if id(node) in reference_counts:
            return reference_counts[id(node)]
        ancestors = ancestors | {id(node)}
        count = int(
            isinstance(node, dict) and node.get("corpus_citation_path") == citation
        )
        children = node.values() if isinstance(node, dict) else node
        total = count + sum(count_references(child, ancestors) for child in children)
        reference_counts[id(node)] = total
        return total

    if count_references(payload) != admitted_reference_count:
        raise ValueError(
            "external source has evidence outside admitted scalar formula proofs"
        )
    return required


def _same_typed_value(left, right) -> bool:
    pending = [(left, right)]
    seen = set()
    while pending:
        left, right = pending.pop()
        if type(left) is not type(right):
            return False
        pair = (id(left), id(right))
        if pair in seen:
            continue
        seen.add(pair)
        if isinstance(left, dict):
            left_keys = {(type(key), key): key for key in left}
            right_keys = {(type(key), key): key for key in right}
            if left_keys.keys() != right_keys.keys():
                return False
            pending.extend(
                (left[left_keys[key]], right[right_keys[key]]) for key in left_keys
            )
        elif isinstance(left, (list, tuple)):
            if len(left) != len(right):
                return False
            pending.extend(zip(left, right, strict=True))
        elif left != right:
            return False
    return True


def external_parameter_preservation_issues(
    payload, admission: dict | None
) -> list[str]:
    if admission is None:
        return []
    required = admission.get("required_unchanged_rules", [])
    if not required:
        return []
    rules = payload.get("rules", []) if isinstance(payload, dict) else []
    if not isinstance(rules, list):
        return ["External source replacement requires a rules list"]
    issues = []
    for original in required:
        matches = [
            r
            for r in rules
            if isinstance(r, dict) and r.get("name") == original["name"]
        ]
        if len(matches) != 1 or not _same_typed_value(matches[0], original):
            issues.append(
                f"External source replacement must preserve the entire legacy scalar rule {original['name']} unchanged, including its source proof, effective period, unit and formula"
            )
    return issues
