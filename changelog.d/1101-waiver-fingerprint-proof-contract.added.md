Contract regression tests pinning that base proof validation stays inside
the waiver-audit fingerprint pipeline: the ci pipeline reports unresolved
proof sources end-to-end under a bound release, and `fingerprint_outcome`
is sensitive to base proof failures. Required by the shared workflow's
waiver-consistent proof-validate skip (TheAxiomFoundation/.github#38).
