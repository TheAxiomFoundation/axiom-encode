//go:build (darwin || linux) && signing_supervisor_test_fixture

package main

// This policy exists only for binaries compiled explicitly with the
// signing_supervisor_test_fixture build tag. Production builds cannot enable it
// at runtime or through an environment variable.
const trustPolicyDescription = "root- or current-user-owned test fixture"
const supervisorBuildKind = "test-fixture-nonpublishable"

func trustPolicyAcceptsOwner(owner uint32, effectiveUser uint32) bool {
	return owner == 0 || owner == effectiveUser
}

func trustPolicyAllowsWritablePath() bool {
	return true
}

func trustPolicyAllowsRootOwnedStickyAncestor(owner uint32) bool {
	return owner == 0
}
