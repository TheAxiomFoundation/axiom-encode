//go:build darwin

package main

import (
	"os"
	"testing"
)

// completeTestBrokerPeerAuthentication is a no-op: the Darwin broker does not
// challenge its peer (harden_darwin.go).
func completeTestBrokerPeerAuthentication(t *testing.T, _ *os.File) {
	t.Helper()
}
