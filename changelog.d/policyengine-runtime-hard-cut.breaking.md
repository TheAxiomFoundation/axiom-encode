Require every PolicyEngine oracle run to use the exact commit in the canonical
RuleSpec checkout's committed `.axiom/policyengine-runtime.toml`. Prove that
commit by a fresh fetch from the code-owned official HTTPS repository, compare
the local checkout with the fetched tree, and execute unprivileged from a
root-owned, caller-nonwritable, self-contained `.venv` closure. Launch admission
and oracle code only through `python -I -S -B`; reject symlinks, writable or
special entries, host Python components, `.pth`, customization hooks, and
`pyvenv.cfg`; and parent-hash the full checkout, standard library, dependency,
metadata, and native-file closure into `axiom-policyengine-runtime/v2` evidence.
Reject runtime mutation or zero comparable evidence, and remove TAXSIM, the
`all` oracle selector, country overrides, ambient interpreter discovery, and
package-install fallbacks from validation and eval workflows.
Remove TAXSIM prediction, run-database, calibration, telemetry, Supabase, and
run-log compatibility fields end to end; persisted payloads using the removed
shape are rejected rather than translated.
