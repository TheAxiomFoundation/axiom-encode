# Reviewer CODEX_HOME read-deny investigation

Codex CLI 0.144.0 supports granular named permission profiles. The reviewer
selects an `axiom-reviewer` profile that starts restricted and grants only:

- `:minimal = "read"`, for executables and runtime libraries required to run a
  shell command; and
- `:workspace_roots = { "." = "read" }`, for the reviewer's working tree.

The profile grants no writes and explicitly disables network access. Unlike the
legacy `--sandbox read-only` mode, it does not grant `:root`/full-disk read.
Codex authentication remains in the parent process: the temporary
`CODEX_HOME/auth.json` is available when Codex starts, while model-issued child
commands receive the restricted profile. Reviewer prompts contain the RuleSpec,
test, review, and oracle context, so the reviewer does not require filesystem
reads to produce its result.

## Deterministic harness

Run this outside an existing macOS seatbelt, with `WORKSPACE` set to the review
cwd and `REVIEWER_HOME` set to a disposable minimal Codex home containing a
non-secret sentinel `auth.json`:

```bash
profile='permissions.axiom-reviewer={filesystem={":minimal"="read",":workspace_roots"={"."="read"}},network={enabled=false}}'
codex=(/opt/axiom-verification/bin/codex -c 'default_permissions="axiom-reviewer"' -c "$profile")

"${codex[@]}" sandbox -P axiom-reviewer -C "$WORKSPACE" -- \
  /bin/sh -c 'head -c 1 pyproject.toml >/dev/null'
"${codex[@]}" sandbox -P axiom-reviewer -C "$WORKSPACE" -- \
  /bin/sh -c 'head -c 1 "$1/auth.json" >/dev/null' sh "$REVIEWER_HOME"
"${codex[@]}" sandbox -P axiom-reviewer -C "$WORKSPACE" -- \
  /usr/bin/python3 -c 'import pathlib,sys; pathlib.Path(sys.argv[1]).read_bytes()' \
  "$REVIEWER_HOME/auth.json"
ln -s "$REVIEWER_HOME/auth.json" "$WORKSPACE/reviewer-auth-link"
"${codex[@]}" sandbox -P axiom-reviewer -C "$WORKSPACE" -- \
  /bin/sh -c 'head -c 1 reviewer-auth-link >/dev/null'
```

The first command must exit 0. The direct shell read, Python read, and
workspace-symlink traversal must all exit nonzero with a sandbox denial. Remove
the disposable symlink after the run.

In the Codex-hosted repair session used for this change, all nested `codex
sandbox` commands stopped at `sandbox-exec: sandbox_apply: Operation not
permitted` (exit 71), because that session was already inside a macOS seatbelt.
This is an outer-environment limitation, not a read-allow result. The profile
was accepted by Codex before seatbelt application. `codex login status` reported
`Logged in using ChatGPT`; the one prompt-only `codex exec` smoke reached the
authenticated Responses endpoint but could not complete because this repair
session also blocked its outbound request.
