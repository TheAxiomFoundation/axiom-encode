//go:build linux

package main

import (
	"io"
	"os"
	"testing"
)

// completeTestBrokerPeerAuthentication answers the Linux broker's peer challenge.
func completeTestBrokerPeerAuthentication(t *testing.T, supervisor *os.File) {
	t.Helper()
	if _, err := supervisor.Write([]byte{brokerPeerChallenge}); err != nil {
		t.Fatal(err)
	}
	response := make([]byte, 1)
	if _, err := io.ReadFull(supervisor, response); err != nil {
		t.Fatal(err)
	}
	if response[0] != brokerPeerResponse {
		t.Fatalf("broker peer response = %#x, want %#x", response[0], brokerPeerResponse)
	}
}
