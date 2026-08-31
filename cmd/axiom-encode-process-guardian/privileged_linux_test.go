//go:build linux

package main

import (
	"os"
	"testing"
)

const requirePrivilegedGuardianTestsEnvironment = "AXIOM_REQUIRE_PRIVILEGED_GUARDIAN_TESTS"

func privilegedGuardianTestsRequired() bool {
	return os.Getenv(requirePrivilegedGuardianTestsEnvironment) == "1"
}

func unavailablePrivilegedGuardianPrerequisite(
	t *testing.T,
	format string,
	arguments ...any,
) {
	t.Helper()
	if privilegedGuardianTestsRequired() {
		t.Fatalf(format, arguments...)
	}
	t.Skipf(format, arguments...)
}
