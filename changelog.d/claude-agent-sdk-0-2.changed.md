Upgrade the `api` extra's `claude-agent-sdk` from the yanked 0.1.47, which PyPI
schedules for deletion, to 0.2.160, and raise its floor to `>=0.2.160` so
`uv sync --locked --extra api` keeps resolving. No encoder module imports the
SDK, so no code changes; the protected verification site-packages, installed
unlocked, already ran the 0.2 line.
