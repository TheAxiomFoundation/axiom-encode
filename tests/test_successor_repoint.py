"""Contract tests for the model-free legacy successor repoint."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import PurePosixPath

import pytest
import yaml

from axiom_encode.successor_repoint import (
    ENVELOPE_SCHEMA,
    RECEIPT_SCHEMA,
    SuccessorRepointError,
    load_repoint_request_bytes,
    load_repoint_request_payload,
    module_identity,
    prove_concept_map,
    receipt_identity_payload,
    receipt_identity_sha256,
    reconcile_money_atom_ratchet,
    reconcile_upstream_source_check_baseline,
    repoint_reference_inventory_issues,
    rewrite_repoint_file,
    scope_module_path,
)

LEGACY_PRIMARY = "us/policies/irs/legacy-table.yaml"
SUCCESSOR_PRIMARY = "us/policies/irs/page-15.yaml"
DEPENDENT_PRIMARY = "us/statutes/26/32.yaml"
LEGACY_IDENTITY = "us:policies/irs/legacy-table"
SUCCESSOR_IDENTITY = "us:policies/irs/page-15"

PAIRS = [
    ("legacy_amounts", "successor_amounts"),
    ("legacy_cap", "successor_cap"),
]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _legacy_module(
    *,
    effective_to: str | None = None,
    indexed_by: str = "child_count",
    values: dict[int, int] | None = None,
    cap: str = "12200",
    extra_rules: list[dict] | None = None,
) -> bytes:
    table_version: dict[str, object] = {"effective_from": "2026-01-01"}
    if effective_to is not None:
        table_version["effective_to"] = effective_to
    table_version["values"] = dict(values or {0: 8680, 1: 13020, 2: 18290, 3: 18290})
    cap_version: dict[str, object] = {"effective_from": "2026-01-01"}
    if effective_to is not None:
        cap_version["effective_to"] = effective_to
    cap_version["formula"] = cap
    payload = {
        "format": "rulespec/v1",
        "module": {"source_verification": {"corpus_citation_path": "us/guidance/x"}},
        "rules": [
            {
                "name": "legacy_amounts",
                "kind": "parameter",
                "dtype": "Money",
                "unit": "USD",
                "indexed_by": indexed_by,
                "versions": [table_version],
            },
            {
                "name": "legacy_cap",
                "kind": "parameter",
                "dtype": "Money",
                "unit": "USD",
                "versions": [cap_version],
            },
            *(extra_rules or []),
        ],
    }
    return yaml.safe_dump(payload, sort_keys=False).encode("utf-8")


def _successor_module(
    *,
    effective_to: str | None = "2026-12-31",
    indexed_by: str = "child_count_category",
    values: dict[int, int] | None = None,
    cap: str = "12200",
    dtype: str = "Money",
    unit: str = "USD",
) -> bytes:
    table_version: dict[str, object] = {"effective_from": "2026-01-01"}
    if effective_to is not None:
        table_version["effective_to"] = effective_to
    table_version["values"] = dict(values or {0: 8680, 1: 13020, 2: 18290, 3: 18290})
    cap_version: dict[str, object] = {"effective_from": "2026-01-01"}
    if effective_to is not None:
        cap_version["effective_to"] = effective_to
    cap_version["formula"] = cap
    payload = {
        "format": "rulespec/v1",
        "module": {
            "source_verification": {"corpus_citation_path": "us/guidance/page-15"}
        },
        "rules": [
            {
                "name": "successor_amounts",
                "kind": "parameter",
                "dtype": dtype,
                "unit": unit,
                "indexed_by": indexed_by,
                "versions": [table_version],
            },
            {
                "name": "successor_cap",
                "kind": "parameter",
                "dtype": "Money",
                "unit": "USD",
                "versions": [cap_version],
            },
        ],
    }
    return yaml.safe_dump(payload, sort_keys=False).encode("utf-8")


DEPENDENT_TEXT = """\
format: rulespec/v1
imports:
  - us:statutes/26/152/c
  - us:policies/irs/legacy-table
module:
  proof_validation:
    required: true
  source_verification:
    corpus_citation_path: us/statute/26/32
  deferred_outputs:
    - output: us:statutes/26/32/f#table_credit
      blocked_by:
        - us:policies/irs/legacy-table#legacy_amounts
      reason: >-
        Subsection (f) directs the Secretary to prescribe tables.
rules:
  - name: capped_child_count
    kind: derived
    entity: TaxUnit
    dtype: Integer
    period: Year
    versions:
      - effective_from: '2026-01-01'
        formula: min(3, child_count)
  - name: earned_income_amount
    kind: derived
    entity: TaxUnit
    dtype: Money
    period: Year
    unit: USD
    metadata:
      proof:
        atoms:
          - path: versions[0].formula
            kind: import
            import:
              target: us:policies/irs/legacy-table#legacy_amounts
              output: legacy_amounts
              hash: sha256:{legacy_digest}
    versions:
      - effective_from: '2026-01-01'
        formula: |-
          match capped_child_count:
              0 => legacy_amounts[0]
              1 => legacy_amounts[1]
              2 => legacy_amounts[2]
              3 => legacy_amounts[3]
  - name: investment_ok
    kind: derived
    entity: TaxUnit
    dtype: Judgment
    period: Year
    metadata:
      proof:
        atoms:
          - path: versions[0].formula
            kind: import
            import:
              target: us:policies/irs/legacy-table#legacy_cap
              output: legacy_cap
              hash: sha256:{legacy_digest}
    versions:
      - effective_from: '2026-01-01'
        formula: investment_income <= legacy_cap
"""


def _dependent(legacy_raw: bytes, *, text: str | None = None) -> bytes:
    return (
        (text or DEPENDENT_TEXT)
        .format(legacy_digest=hashlib.sha256(legacy_raw).hexdigest())
        .encode("utf-8")
    )


def _envelope(**overrides) -> dict:
    payload = {
        "schema": ENVELOPE_SCHEMA,
        "legacy_primary": LEGACY_PRIMARY,
        "successor_primary": SUCCESSOR_PRIMARY,
        "dependents": [DEPENDENT_PRIMARY],
        "concept_map": [{"from": old, "to": new} for old, new in PAIRS],
        "program_scope_updates": [
            {"program_spec": "programs/us/fiit/fy-2026.yaml", "scope": "federal"}
        ],
    }
    payload.update(overrides)
    return payload


def _prove(legacy: bytes, successor: bytes, dependent: bytes, **overrides):
    request = load_repoint_request_payload(_envelope(**overrides))
    return request, prove_concept_map(
        legacy_raw=legacy,
        successor_raw=successor,
        request=request,
        dependent_raws={DEPENDENT_PRIMARY: dependent},
    )


# ---------------------------------------------------------------------------
# Envelope
# ---------------------------------------------------------------------------


class TestEnvelope:
    def test_parses_the_canonical_envelope(self):
        request = load_repoint_request_payload(_envelope())
        assert request.legacy_identity == LEGACY_IDENTITY
        assert request.successor_identity == SUCCESSOR_IDENTITY
        assert request.legacy_companion == PurePosixPath(
            "us/policies/irs/legacy-table.test.yaml"
        )
        assert request.legacy_scope_path == "policies/irs/legacy-table"
        assert request.successor_scope_path == "policies/irs/page-15"
        assert request.concept_renames == dict(PAIRS)
        assert request.sha256 == hashlib.sha256(request.canonical_bytes).hexdigest()

    def test_round_trips_through_bytes(self):
        raw = json.dumps(_envelope()).encode("utf-8")
        assert load_repoint_request_bytes(raw).sha256 == (
            load_repoint_request_payload(_envelope()).sha256
        )

    @pytest.mark.parametrize(
        "overrides, message",
        [
            ({"schema": "axiom-encode/other/v1"}, "schema must be"),
            ({"dependents": []}, "dependents must be a non-empty array"),
            ({"dependents": [DEPENDENT_PRIMARY, DEPENDENT_PRIMARY]}, "must be unique"),
            ({"dependents": [LEGACY_PRIMARY]}, "cannot name the legacy"),
            ({"dependents": ["us/statutes/26/32.test.yaml"]}, "protected primary"),
            ({"dependents": ["us-ca/statutes/26/32.yaml"]}, "legacy jurisdiction"),
            ({"dependents": ["/abs/us/statutes/26/32.yaml"]}, "protected primary"),
            ({"dependents": ["us/programs/x.yaml"]}, "protected primary"),
            ({"successor_primary": LEGACY_PRIMARY}, "primaries must differ"),
            ({"concept_map": []}, "concept_map must be a non-empty array"),
            (
                {"concept_map": [{"from": "a", "to": "a"}]},
                "no-op rename",
            ),
            (
                {"concept_map": [{"from": "a", "to": "b"}, {"from": "a", "to": "c"}]},
                "from is duplicated",
            ),
            (
                {"concept_map": [{"from": "a", "to": "c"}, {"from": "b", "to": "c"}]},
                "collides with another rename",
            ),
            (
                {"concept_map": [{"from": "a", "to": "b"}, {"from": "b", "to": "c"}]},
                "chain or swap",
            ),
            ({"concept_map": [{"from": "A", "to": "b"}]}, "lowercase concept"),
            ({"concept_map": [{"from": "a"}]}, "exactly from and to"),
            (
                {
                    "program_scope_updates": [
                        {"program_spec": "elsewhere/x.yaml", "scope": "a"}
                    ]
                },
                "programs/",
            ),
            (
                {
                    "program_scope_updates": [
                        {"program_spec": "programs/us/a.yaml", "scope": "Federal"}
                    ]
                },
                "lowercase identifier",
            ),
        ],
    )
    def test_refuses_malformed_envelopes(self, overrides, message):
        with pytest.raises(SuccessorRepointError, match=message):
            load_repoint_request_payload(_envelope(**overrides))

    def test_refuses_extra_and_missing_fields(self):
        extra = _envelope()
        extra["extra"] = 1
        with pytest.raises(SuccessorRepointError, match="exactly"):
            load_repoint_request_payload(extra)
        missing = _envelope()
        del missing["program_scope_updates"]
        with pytest.raises(SuccessorRepointError, match="exactly"):
            load_repoint_request_payload(missing)

    def test_refuses_an_oversize_envelope(self):
        with pytest.raises(SuccessorRepointError, match="64 KiB"):
            load_repoint_request_bytes(b"{" + b" " * (64 * 1024) + b"}")

    def test_refuses_non_json(self):
        with pytest.raises(SuccessorRepointError, match="not valid UTF-8 JSON"):
            load_repoint_request_bytes(b"not json")


# ---------------------------------------------------------------------------
# Concept equivalence proof
# ---------------------------------------------------------------------------


class TestConceptProof:
    def test_proves_indexed_tables_and_scalars(self):
        legacy = _legacy_module()
        successor = _successor_module()
        _request, proofs = _prove(legacy, successor, _dependent(legacy))
        assert proofs.successor_window_start == "2026-01-01"
        assert proofs.successor_window_end == "2026-12-31"
        assert proofs.renames == dict(PAIRS)
        table, cap = proofs.proofs
        assert table.keys == (0, 1, 2, 3)
        assert table.indexed_by_from == "child_count"
        assert table.indexed_by_to == "child_count_category"
        assert table.literal_subscripts == (0, 1, 2, 3)
        assert table.formula_uses == 4
        assert table.reference_uses == 2  # proof atom target plus blocked_by
        assert cap.keys == (0,)
        assert cap.formula_uses == 1
        assert cap.literal_subscripts == ()
        assert "2026-01-01" in table.probes and "2026-12-31" in table.probes

    def test_records_the_post_window_behaviour_change(self):
        legacy = _legacy_module()
        _request, proofs = _prove(legacy, _successor_module(), _dependent(legacy))
        assert proofs.post_window_behavior_change is True
        assert all(
            item["extends_past_successor_window"] is True
            for item in proofs.dependent_use_windows
        )

    def test_no_behaviour_change_when_the_successor_window_covers_the_use(self):
        legacy = _legacy_module()
        dependent = (
            _dependent(legacy)
            .replace(
                b"      - effective_from: '2026-01-01'\n        formula: |-",
                b"      - effective_from: '2026-01-01'\n"
                b"        effective_to: '2026-12-31'\n        formula: |-",
            )
            .replace(
                b"      - effective_from: '2026-01-01'\n"
                b"        formula: investment_income <= legacy_cap",
                b"      - effective_from: '2026-01-01'\n"
                b"        effective_to: '2026-12-31'\n"
                b"        formula: investment_income <= legacy_cap",
            )
        )
        _request, proofs = _prove(legacy, _successor_module(), dependent)
        assert proofs.post_window_behavior_change is False

    def test_refuses_a_value_mismatch(self):
        legacy = _legacy_module()
        successor = _successor_module(values={0: 8680, 1: 13020, 2: 18290, 3: 99999})
        with pytest.raises(SuccessorRepointError, match=r"differs at 2026-01-01"):
            _prove(legacy, successor, _dependent(legacy))

    def test_refuses_a_key_set_mismatch(self):
        legacy = _legacy_module()
        successor = _successor_module(values={0: 8680, 1: 13020, 2: 18290})
        with pytest.raises(SuccessorRepointError, match="different table keys"):
            _prove(legacy, successor, _dependent(legacy))

    def test_refuses_a_scalar_value_mismatch(self):
        legacy = _legacy_module()
        successor = _successor_module(cap="12300")
        with pytest.raises(SuccessorRepointError, match="differs at 2026-01-01"):
            _prove(legacy, successor, _dependent(legacy))

    def test_refuses_a_surface_mismatch(self):
        legacy = _legacy_module()
        successor = _successor_module(unit="EUR")
        with pytest.raises(SuccessorRepointError, match="differs on unit"):
            _prove(legacy, successor, _dependent(legacy))

    def test_refuses_a_scalar_to_table_change(self):
        legacy = _legacy_module()
        successor = yaml.safe_load(_successor_module().decode())
        successor["rules"][1]["indexed_by"] = "child_count_category"
        successor["rules"][1]["versions"][0].pop("formula")
        successor["rules"][1]["versions"][0]["values"] = {0: 12200}
        with pytest.raises(
            SuccessorRepointError, match="scalar parameter and an indexed table"
        ):
            _prove(
                legacy,
                yaml.safe_dump(successor).encode("utf-8"),
                _dependent(legacy),
            )

    def test_indexed_by_rename_requires_literal_subscripts(self):
        legacy = _legacy_module()
        dependent = _dependent(legacy).replace(
            b"0 => legacy_amounts[0]", b"0 => legacy_amounts[capped_child_count]"
        )
        with pytest.raises(
            SuccessorRepointError, match="without a literal integer subscript"
        ):
            _prove(legacy, _successor_module(), dependent)

    def test_indexed_by_rename_requires_subscripts_the_successor_defines(self):
        legacy = _legacy_module(values={0: 1, 1: 2, 2: 3, 3: 4, 7: 5})
        successor = _successor_module(values={0: 1, 1: 2, 2: 3, 3: 4, 7: 5})
        dependent = _dependent(legacy).replace(
            b"3 => legacy_amounts[3]", b"3 => legacy_amounts[9]"
        )
        with pytest.raises(SuccessorRepointError, match=r"subscripted with keys \[9\]"):
            _prove(legacy, successor, dependent)

    def test_matching_indexed_by_admits_a_non_literal_subscript(self):
        legacy = _legacy_module(indexed_by="child_count_category")
        dependent = _dependent(legacy).replace(
            b"0 => legacy_amounts[0]", b"0 => legacy_amounts[capped_child_count]"
        )
        _request, proofs = _prove(legacy, _successor_module(), dependent)
        assert proofs.proofs[0].indexed_by_from == "child_count_category"

    def test_refuses_a_legacy_gap_inside_the_successor_window(self):
        legacy = yaml.safe_load(_legacy_module().decode())
        legacy["rules"][0]["versions"] = [
            {
                "effective_from": "2026-01-01",
                "effective_to": "2026-03-31",
                "values": {0: 8680, 1: 13020, 2: 18290, 3: 18290},
            },
            {
                "effective_from": "2026-07-01",
                "values": {0: 8680, 1: 13020, 2: 18290, 3: 18290},
            },
        ]
        raw = yaml.safe_dump(legacy).encode("utf-8")
        with pytest.raises(SuccessorRepointError, match="gap or overlap"):
            _prove(raw, _successor_module(), _dependent(raw))

    def test_refuses_an_open_successor_window_the_legacy_does_not_cover(self):
        legacy = _legacy_module(effective_to="2026-06-30")
        successor = _successor_module(effective_to=None)
        with pytest.raises(SuccessorRepointError, match="open successor window"):
            _prove(legacy, successor, _dependent(legacy))

    def test_refuses_concepts_the_modules_do_not_export(self):
        legacy = _legacy_module()
        with pytest.raises(
            SuccessorRepointError, match="legacy module does not export"
        ):
            _prove(
                legacy,
                _successor_module(),
                _dependent(legacy),
                concept_map=[{"from": "missing", "to": "successor_cap"}],
            )
        with pytest.raises(
            SuccessorRepointError, match="successor module does not export"
        ):
            _prove(
                legacy,
                _successor_module(),
                _dependent(legacy),
                concept_map=[{"from": "legacy_cap", "to": "missing"}],
            )

    def test_refuses_an_unmapped_legacy_reference(self):
        legacy = _legacy_module()
        with pytest.raises(SuccessorRepointError, match="unmapped legacy concept"):
            _prove(
                legacy,
                _successor_module(),
                _dependent(legacy),
                concept_map=[{"from": "legacy_amounts", "to": "successor_amounts"}],
            )

    def test_refuses_a_duplicate_export(self):
        legacy = yaml.safe_load(_legacy_module().decode())
        legacy["rules"].append(copy.deepcopy(legacy["rules"][0]))
        raw = yaml.safe_dump(legacy).encode("utf-8")
        with pytest.raises(SuccessorRepointError, match="duplicate concept"):
            _prove(raw, _successor_module(), _dependent(raw))

    def test_refuses_a_dependent_that_already_defines_the_successor_concept(self):
        legacy = _legacy_module()
        dependent = _dependent(legacy).replace(
            b"  - name: capped_child_count", b"  - name: successor_cap"
        )
        with pytest.raises(SuccessorRepointError, match="already defines successor"):
            _prove(legacy, _successor_module(), dependent)

    def test_refuses_a_dependent_that_already_imports_the_successor(self):
        legacy = _legacy_module()
        dependent = _dependent(legacy).replace(
            b"  - us:statutes/26/152/c", b"  - us:policies/irs/page-15"
        )
        with pytest.raises(
            SuccessorRepointError, match="already imports the successor"
        ):
            _prove(legacy, _successor_module(), dependent)

    def test_refuses_successor_concepts_with_different_windows(self):
        successor = yaml.safe_load(_successor_module().decode())
        successor["rules"][1]["versions"][0]["effective_to"] = "2026-06-30"
        legacy = _legacy_module()
        with pytest.raises(SuccessorRepointError, match="one validity window"):
            _prove(
                legacy,
                yaml.safe_dump(successor).encode("utf-8"),
                _dependent(legacy),
            )

    def test_refuses_a_successor_whose_kind_differs(self):
        successor = yaml.safe_load(_successor_module().decode())
        successor["rules"][1]["kind"] = "derived"
        legacy = _legacy_module()
        with pytest.raises(SuccessorRepointError, match="differs on kind"):
            _prove(
                legacy,
                yaml.safe_dump(successor).encode("utf-8"),
                _dependent(legacy),
            )

    def test_refuses_a_derived_concept_pair(self):
        legacy = yaml.safe_load(_legacy_module().decode())
        successor = yaml.safe_load(_successor_module().decode())
        legacy["rules"][1]["kind"] = "derived"
        successor["rules"][1]["kind"] = "derived"
        legacy_raw = yaml.safe_dump(legacy).encode("utf-8")
        with pytest.raises(SuccessorRepointError, match="not kind: parameter"):
            _prove(
                legacy_raw,
                yaml.safe_dump(successor).encode("utf-8"),
                _dependent(legacy_raw),
            )


# ---------------------------------------------------------------------------
# Exact-token rewrite
# ---------------------------------------------------------------------------


def _rewrite(dependent: bytes, successor: bytes, *, renames=None, primary=True):
    return rewrite_repoint_file(
        dependent,
        primary=primary,
        legacy_identity=LEGACY_IDENTITY,
        successor_identity=SUCCESSOR_IDENTITY,
        successor_sha256=hashlib.sha256(successor).hexdigest(),
        renames=dict(renames or PAIRS),
        label=DEPENDENT_PRIMARY,
    )


class TestRewrite:
    def test_rewrites_every_authorised_surface(self):
        legacy = _legacy_module()
        successor = _successor_module()
        out, replacements = _rewrite(_dependent(legacy), successor)
        text = out.decode("utf-8")
        payload = yaml.safe_load(text)
        assert payload["imports"] == ["us:statutes/26/152/c", SUCCESSOR_IDENTITY]
        assert payload["module"]["deferred_outputs"][0]["blocked_by"] == [
            f"{SUCCESSOR_IDENTITY}#successor_amounts"
        ]
        atom = payload["rules"][1]["metadata"]["proof"]["atoms"][0]["import"]
        assert atom["target"] == f"{SUCCESSOR_IDENTITY}#successor_amounts"
        assert atom["output"] == "successor_amounts"
        assert atom["hash"] == f"sha256:{hashlib.sha256(successor).hexdigest()}"
        assert (
            "0 => successor_amounts[0]" in payload["rules"][1]["versions"][0]["formula"]
        )
        assert "investment_income <= successor_cap" in text
        assert LEGACY_IDENTITY not in text
        assert "legacy_amounts" not in text
        operations = {item["operation"] for item in replacements if "operation" in item}
        assert operations == {"retarget_proof_import_hash"}
        assert any(
            item.get("from") == LEGACY_IDENTITY and item.get("to") == SUCCESSOR_IDENTITY
            for item in replacements
        )

    def test_preserves_every_byte_outside_the_rewritten_tokens(self):
        legacy = _legacy_module()
        before = _dependent(legacy)
        out, _ = _rewrite(before, _successor_module())
        # Only the mapped tokens and the retargeted hash may differ.
        assert before.count(b"\n") == out.count(b"\n")
        assert b"Subsection (f) directs the Secretary to prescribe tables." in out
        assert b"min(3, child_count)" in out

    def test_leaves_quoted_substrings_inside_a_formula_untouched(self):
        legacy = _legacy_module()
        dependent = _dependent(legacy).replace(
            b"        formula: investment_income <= legacy_cap",
            b'        formula: label == "legacy_cap" and investment_income <= legacy_cap',
        )
        out, _ = _rewrite(dependent, _successor_module())
        text = out.decode("utf-8")
        assert 'label == "legacy_cap"' in text
        assert "investment_income <= successor_cap" in text

    def test_respects_identifier_boundaries(self):
        legacy = _legacy_module()
        dependent = _dependent(legacy).replace(
            b"        formula: investment_income <= legacy_cap",
            b"        formula: legacy_cap_extra + legacy_cap + x_legacy_cap",
        )
        out, _ = _rewrite(dependent, _successor_module())
        text = out.decode("utf-8")
        assert "legacy_cap_extra + successor_cap + x_legacy_cap" in text

    def test_refuses_a_legacy_concept_named_outside_a_rewritable_surface(self):
        legacy = _legacy_module()
        dependent = _dependent(legacy).replace(
            b"      reason: >-\n        Subsection (f) directs the Secretary to prescribe tables.",
            b"      reason: >-\n        Blocked on legacy_cap until the table lands.",
        )
        with pytest.raises(SuccessorRepointError, match="outside a rewritable surface"):
            _rewrite(dependent, _successor_module())

    def test_refuses_the_legacy_identity_outside_a_rewritable_surface(self):
        legacy = _legacy_module()
        dependent = _dependent(legacy).replace(
            b"    corpus_citation_path: us/statute/26/32",
            b"    corpus_citation_path: us/statute/26/32\n"
            b"    note: us:policies/irs/legacy-table",
        )
        with pytest.raises(SuccessorRepointError, match="outside a rewritable surface"):
            _rewrite(dependent, _successor_module())

    def test_refuses_a_quoted_yaml_formula_scalar(self):
        legacy = _legacy_module()
        dependent = _dependent(legacy).replace(
            b"        formula: investment_income <= legacy_cap",
            b"        formula: 'investment_income <= legacy_cap'",
        )
        with pytest.raises(SuccessorRepointError, match="quoted YAML scalar"):
            _rewrite(dependent, _successor_module())

    def test_refuses_a_yaml_alias(self):
        legacy = _legacy_module()
        dependent = _dependent(legacy).replace(
            b"imports:\n  - us:statutes/26/152/c",
            b"anchors: &a us:policies/irs/legacy-table\nimports:\n  - *a",
        )
        with pytest.raises(SuccessorRepointError, match="anchor or alias"):
            _rewrite(dependent, _successor_module())

    def test_refuses_a_file_with_no_legacy_reference(self):
        legacy = _legacy_module()
        clean = (
            _dependent(legacy)
            .replace(LEGACY_IDENTITY.encode(), b"us:policies/irs/other")
            .replace(b"legacy_amounts", b"other_amounts")
            .replace(b"legacy_cap", b"other_cap")
        )
        with pytest.raises(SuccessorRepointError, match="no legacy reference"):
            _rewrite(clean, _successor_module())

    def test_refuses_an_unmapped_fragment(self):
        legacy = _legacy_module()
        with pytest.raises(SuccessorRepointError, match="unmapped legacy concept"):
            _rewrite(
                _dependent(legacy),
                _successor_module(),
                renames=[("legacy_cap", "successor_cap")],
            )

    def test_refuses_a_proof_atom_without_a_hash(self):
        legacy = _legacy_module()
        dependent = _dependent(legacy).replace(
            b"              hash: sha256:"
            + hashlib.sha256(legacy).hexdigest().encode()
            + b"\n",
            b"",
            1,
        )
        with pytest.raises(SuccessorRepointError, match="no hash to retarget"):
            _rewrite(dependent, _successor_module())

    def test_refuses_an_invalid_successor_digest(self):
        with pytest.raises(SuccessorRepointError, match="SHA-256 hex"):
            rewrite_repoint_file(
                _dependent(_legacy_module()),
                primary=True,
                legacy_identity=LEGACY_IDENTITY,
                successor_identity=SUCCESSOR_IDENTITY,
                successor_sha256="short",
                renames=dict(PAIRS),
                label=DEPENDENT_PRIMARY,
            )

    def test_rewrites_companion_test_mapping_keys(self):
        companion = (
            "cases:\n"
            "  - outputs:\n"
            f"      {LEGACY_IDENTITY}#legacy_cap: 12200\n"
            "      us:statutes/26/32#earned_income_amount: 8680\n"
        ).encode("utf-8")
        out, replacements = _rewrite(companion, _successor_module(), primary=False)
        payload = yaml.safe_load(out.decode("utf-8"))
        assert f"{SUCCESSOR_IDENTITY}#successor_cap" in payload["cases"][0]["outputs"]
        assert replacements[0]["count"] == 1


# ---------------------------------------------------------------------------
# Reference inventory
# ---------------------------------------------------------------------------


class TestReferenceInventory:
    def _files(self, **extra) -> dict[str, bytes]:
        legacy = _legacy_module()
        files = {
            LEGACY_PRIMARY: legacy,
            SUCCESSOR_PRIMARY: _successor_module(),
            DEPENDENT_PRIMARY: _dependent(legacy),
            "programs/us/fiit/fy-2026.yaml": (
                "program: us/fiit\nscope:\n  federal:\n"
                "    - policies/irs/legacy-table\n"
            ).encode("utf-8"),
            "us/statutes/26/24/d.yaml": b"imports:\n  - us:statutes/26/32\n",
        }
        files.update(extra)
        return files

    def test_accepts_a_fully_declared_inventory(self):
        request = load_repoint_request_payload(_envelope())
        assert repoint_reference_inventory_issues(self._files(), request=request) == []

    def test_fails_closed_on_an_undeclared_protected_referrer(self):
        request = load_repoint_request_payload(_envelope())
        files = self._files(
            **{"us/statutes/26/99.yaml": f"imports:\n  - {LEGACY_IDENTITY}\n".encode()}
        )
        issues = repoint_reference_inventory_issues(files, request=request)
        assert issues == [
            "us/statutes/26/99.yaml references the legacy module but is not a "
            "declared dependent"
        ]

    def test_fails_closed_on_a_path_form_reference(self):
        request = load_repoint_request_payload(_envelope())
        files = self._files(
            **{"us/statutes/26/99.yaml": f"note: {LEGACY_PRIMARY}\n".encode()}
        )
        assert repoint_reference_inventory_issues(files, request=request)

    def test_fails_closed_on_an_undeclared_program_spec(self):
        request = load_repoint_request_payload(_envelope(program_scope_updates=[]))
        issues = repoint_reference_inventory_issues(self._files(), request=request)
        assert issues == [
            "programs/us/fiit/fy-2026.yaml lists the legacy module but is not a "
            "declared program_scope_updates entry"
        ]

    def test_fails_closed_on_an_untracked_declared_dependent(self):
        request = load_repoint_request_payload(_envelope())
        files = self._files()
        del files[DEPENDENT_PRIMARY]
        issues = repoint_reference_inventory_issues(files, request=request)
        assert issues == [
            f"declared dependent is not tracked at clean HEAD: {DEPENDENT_PRIMARY}"
        ]

    def test_fails_closed_on_an_untracked_declared_program_spec(self):
        request = load_repoint_request_payload(_envelope())
        files = self._files()
        del files["programs/us/fiit/fy-2026.yaml"]
        issues = repoint_reference_inventory_issues(files, request=request)
        assert issues == [
            "declared ProgramSpec is not tracked at clean HEAD: "
            "programs/us/fiit/fy-2026.yaml"
        ]

    def test_ignores_unrelated_files(self):
        request = load_repoint_request_payload(_envelope())
        files = self._files(**{"docs/notes.md": LEGACY_IDENTITY.encode()})
        assert repoint_reference_inventory_issues(files, request=request) == []


# ---------------------------------------------------------------------------
# Metadata reconciliations
# ---------------------------------------------------------------------------


class TestMetadataReconciliation:
    BASELINE = (
        "# header\n"
        "us/policies/irs/alpha.yaml\n"
        "us/policies/irs/legacy-table.yaml\n"
        "us/policies/irs/zeta.yaml\n"
    ).encode("utf-8")

    def test_removes_the_upstream_baseline_entry(self):
        out, operations = reconcile_upstream_source_check_baseline(
            self.BASELINE, legacy_path=LEGACY_PRIMARY
        )
        assert out.decode("utf-8").splitlines() == [
            "# header",
            "us/policies/irs/alpha.yaml",
            "us/policies/irs/zeta.yaml",
        ]
        assert operations[0]["count"] == 1

    def test_refuses_a_missing_or_duplicated_baseline_entry(self):
        with pytest.raises(SuccessorRepointError, match="exactly once"):
            reconcile_upstream_source_check_baseline(
                b"# header\nus/policies/irs/alpha.yaml\n", legacy_path=LEGACY_PRIMARY
            )
        with pytest.raises(SuccessorRepointError, match="exactly once"):
            reconcile_upstream_source_check_baseline(
                self.BASELINE + f"{LEGACY_PRIMARY}\n".encode(),
                legacy_path=LEGACY_PRIMARY,
            )

    RATCHET = (
        "# Money-atom proof-obligation ratchet.\n"
        "total_allowed: 151\n"
        "# Current per-file backlog (informational):\n"
        "#   us/policies/irs/alpha.yaml: 12\n"
        "#   us/policies/irs/legacy-table.yaml: 7\n"
        "#   us/policies/irs/zeta.yaml: 4\n"
    ).encode("utf-8")

    def test_removes_the_money_atom_backlog_comment(self):
        out, operations = reconcile_money_atom_ratchet(
            self.RATCHET, legacy_path=LEGACY_PRIMARY
        )
        assert LEGACY_PRIMARY not in out.decode("utf-8")
        assert yaml.safe_load(out.decode("utf-8")) == {"total_allowed": 151}
        assert operations[0]["count"] == 1

    def test_refuses_when_the_entry_is_not_a_comment(self):
        raw = self.RATCHET.replace(
            b"#   us/policies/irs/legacy-table.yaml: 7",
            b'"us/policies/irs/legacy-table.yaml": 7',
        )
        with pytest.raises(SuccessorRepointError, match="exactly one informational"):
            reconcile_money_atom_ratchet(raw, legacy_path=LEGACY_PRIMARY)

    def test_refuses_a_missing_backlog_entry(self):
        with pytest.raises(SuccessorRepointError, match="exactly one informational"):
            reconcile_money_atom_ratchet(
                b"total_allowed: 151\n", legacy_path=LEGACY_PRIMARY
            )


# ---------------------------------------------------------------------------
# Receipt identity
# ---------------------------------------------------------------------------


class TestReceiptIdentity:
    def _payload(self, **overrides):
        payload = {
            "request_sha256": "a" * 64,
            "base_commit": "b" * 40,
            "base_tree": "c" * 40,
            "legacy_manifest_sha256": "d" * 64,
            "successor_manifest_sha256": "e" * 64,
            "legacy_files": [{"path": LEGACY_PRIMARY, "sha256": "f" * 64}],
            "successor_files": [{"path": SUCCESSOR_PRIMARY, "sha256": "0" * 64}],
            "dependents": [{"primary": DEPENDENT_PRIMARY}],
            "concept_proofs": [{"from": "a", "to": "b"}],
            "metadata_reconciliations": [],
            "program_scope_reconciliations": [],
            "semantics": {"post_window_behavior_change": True},
        }
        payload.update(overrides)
        return receipt_identity_payload(**payload)

    def test_is_deterministic_and_schema_bound(self):
        first = self._payload()
        assert first["schema"] == RECEIPT_SCHEMA
        assert receipt_identity_sha256(first) == receipt_identity_sha256(
            self._payload()
        )

    def test_changes_with_any_bound_field(self):
        baseline = receipt_identity_sha256(self._payload())
        assert (
            receipt_identity_sha256(
                self._payload(semantics={"post_window_behavior_change": False})
            )
            != baseline
        )
        assert receipt_identity_sha256(self._payload(base_commit="9" * 40)) != baseline


def test_module_identity_and_scope_path_helpers():
    assert module_identity(PurePosixPath(LEGACY_PRIMARY)) == LEGACY_IDENTITY
    assert scope_module_path(PurePosixPath(SUCCESSOR_PRIMARY)) == "policies/irs/page-15"
