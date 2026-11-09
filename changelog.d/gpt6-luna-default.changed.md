`encode`, `eval`, and the Codex reviewer now default to `gpt-6-luna`, and
`encode` escalates validator-rejected sections to `gpt-6-sol`. GPT-6 models get
extended output and explicit prompt-cache breakpoints, pricing rates v4 add both
models with vendor provenance, and repair-candidate extraction prefers the GPT-6
Sol escalation candidate while still reading older GPT-5.6 Sol artifacts. When
ChatGPT-account Codex rejects a model it does not serve, the error now names the
explicit-model workaround. The pinned signed-apply lane is unchanged until its
`AXIOM_ENCODE_REF` is re-pinned to a commit with GPT-6 support.
