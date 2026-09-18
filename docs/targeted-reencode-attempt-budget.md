# Targeted re-encode attempt limits

The ad hoc re-encode guard blocks after three consecutive failed model-bearing
runs by default. Triage the actual failure before granting another attempt;
regeneration alone does not fix a validator or source defect. Queue dispatches,
authenticated repair replays and existing repository-wide overrides retain their
established behavior. This is a cost limit, not a signing or validation waiver.

For a reviewed retry that must leave concurrent citations at their existing
limits, set the repository Actions variable `ATTEMPT_BUDGET_BY_CITATION_JSON`
to an object keyed by exact, case-sensitive corpus citation paths. For example:

```json
{
  "de/statute/estg/78": {
    "budget": 4,
    "expires_at": "2026-09-10T00:00:00Z",
    "reason": "Reviewed dependency-link fix in axiom-encode#1588"
  }
}
```

The budget is a positive JSON integer, the expiration is an ISO datetime with a
timezone, and the triage reason is a nonempty string. Set an appropriate future
expiration when applying the example. A limit of four allows a fourth attempt
following three consecutive failures; a fourth failure blocks again. It does
not put the selected citation into report-only mode. Unlisted citations,
including descendants or similar spellings, keep the repository-wide limit.

At the expiration instant the selected entry stops applying. Malformed selected
entries or malformed JSON retain the repository-wide limit with a warning;
unrelated entries do not affect a citation. Remove temporary entries after the
reviewed retry. Preserve other operators' entries when changing the variable.

`ATTEMPT_BUDGET` still sets the default numeric limit. The existing
`ATTEMPT_BUDGET_OVERRIDE=true` remains a repository-wide report-only switch;
it takes precedence over numeric limits and should not be used for an isolated
retry while another citation is being worked on. All protected environment,
source, behavior, completeness and signature gates remain in force.
