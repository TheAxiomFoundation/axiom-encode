//go:build (darwin || linux) && !signing_supervisor_test_fixture

package main

const trustPolicyDescription = "root-owned and not writable by the invoking user"
const supervisorBuildKind = "production"

func trustPolicyAcceptsOwner(owner uint32, _ uint32) bool {
	return owner == 0
}

func trustPolicyAllowsWritablePath() bool {
	return false
}

func trustPolicyAllowsRootOwnedStickyAncestor(_ uint32) bool {
	return false
}
