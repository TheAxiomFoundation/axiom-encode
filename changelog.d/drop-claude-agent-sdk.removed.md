Remove `claude-agent-sdk` from the `api` extra. Nothing has imported it since
the RuleSpec migration deleted `SDKOrchestrator`, and every `.[api]` install
pulled a 103 MB wheel whose bundled 242 MB `claude` binary was copied into the
protected verification runtime. The extra now installs only `anthropic`, and
the lock drops the SDK and the 14 packages only it required.
