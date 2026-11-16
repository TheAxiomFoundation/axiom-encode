# Explicit companion relation edges

A `.test.yaml` case may opt into explicit graph edges with a `relations` list.
The case must provide a nonempty string `entity_id` and assert a derived output;
the derived output's entity is the query root's kind. An empty list still opts
in. `null`, parameter-only cases, and implicit root IDs are rejected.

```yaml
entity_id: claimant-A
relations:
  - name: ca:policies/example#relation.claims
    tuple: [claim-A, claimant-A]
tables:
  Claim:
    - id: claim-A
      ca:policies/example#input.paid: 100
```

Each entry contains exactly `name` and `tuple`. The canonical name must identify
a compiled data relation, with explicit slot kinds and matching arity. Derived
relations cannot be supplied. Endpoints must name the root or table rows with
explicit nonempty string IDs (`id`, `entity_id`, or the existing entity-specific
ID field). Multiple ID fields must agree. IDs cannot belong to different kinds.
Generated anonymous row IDs are not addressable. Edges inherit the case interval;
no per-edge intervals or formulas are accepted. Identical repeated edges retain
the engine's ordinary deduplication behavior.

Both CLI tests and pipeline tests use the same validator. Cases without
`relations` retain their existing identity and compact-row behavior. Existing
required-request anchor schemas are unchanged. Tests-only repair preserves the
entire graph and rejects edge changes to retained cases.

This carrier executes supplied facts; it does not authenticate their completeness
or truth. It does not broaden source-bound execution evidence: the current
`ExecutionCapture` rejects native traces containing non-root entities. A passing
nested graph test therefore does not establish source-bound admission evidence.
