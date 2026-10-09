Raise the corpus release object byte cap from 16 MiB to 64 MiB in the
resolver and the registry materializer, and apply the same cap to every
release-object fetch: the five workflow curl steps and `axiom-encode ci`.
The 25.2 MiB pretty-printed size was reported for the
`us-rulespec-2026-09-14-wave4-r2-union` object, giving approximately 2.5x
headroom at that reported size.
