Add `axiom-encode ci` as a verification-only local entrypoint that resolves a
lane's pinned validate-rulespec toolchain and runs the shared workflow gates in
CI order with matching dependency and changed-file semantics.
