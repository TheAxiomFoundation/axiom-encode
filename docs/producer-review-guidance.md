# Pinned producer review guidance

The targeted signed re-encode workflow accepts a primary `review_finding` starting exactly with `axiom-producer-review-guidance-ref:v1:` followed by a JSON descriptor. Other inline findings retain their existing behavior. Unknown versions in this reserved namespace refuse.

The descriptor has exactly these fields:

```json
{
  "schema": "axiom-encode/producer-review-finding-ref/v1",
  "producer_commit": "<40 lowercase hex characters>",
  "path": "review-guidance/example.json",
  "sha256": "<64 lowercase hex characters>",
  "size_bytes": 1234
}
```

The referenced regular, non-executable Git blob must be in the exact executing producer commit. Its closed JSON envelope contains `schema` (`axiom-encode/producer-review-guidance/v1`), `citation`, `source_sha256`, and `finding`. The citation and source digest must match the existing authenticated canonical source resolution. The finding has no terminal newline; the reader adds exactly one LF. The envelope is limited to 512 KiB and the descriptor to 4096 bytes. These byte limits do not establish model token capacity.

The protected encode caller supplies paired `--producer-review-guidance-ref` and `--producer-review-guidance-checkout` arguments, verifies the checkout against its own attested producer identity, and uses the normal corpus authentication path. Immutable Git reads use the existing protected wrapper; tree metadata is limited after capture to 8 MiB and 100,000 records. This is not a subprocess memory bound.

Only ordinary primary, complete-source, OpenAI apply runs are supported. Canonical replacement is supported. Queue, repair, legacy, dependent, retained-successor and mixed inline/reference routes refuse. No additional source-generation or canonical-refresh lanes are admitted. An empty full-source transaction, including supported corpus/provision transport wrappers, is allowed.

The existing `guard-generated` command receives the original descriptor and independently reconstructs its binding from the successfully verified applied manifest and context. Packaging compares the original workflow request, that successful guard result, and the exact context/manifest buffers written to the artifact. Missing or altered evidence refuses. This feature neither signs diagnostic copies nor makes a candidate publishable by itself; all existing source, runtime, review, signature and publication requirements remain.
