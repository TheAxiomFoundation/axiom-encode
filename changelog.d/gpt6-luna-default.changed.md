`encode` now defaults to `gpt-6-luna` and escalates validator-rejected sections to
`gpt-6-sol`. GPT-6 models get extended output and explicit prompt-cache breakpoints,
pricing rates v4 add both models with vendor provenance, the signed-apply
`initial-model` allowlist accepts them, and repair-candidate extraction prefers the
GPT-6 Sol escalation candidate while still reading older GPT-5.6 Sol artifacts.
