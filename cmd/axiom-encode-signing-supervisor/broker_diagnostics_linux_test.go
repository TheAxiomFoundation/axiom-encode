//go:build linux

package main

import "testing"

func TestBrokerReportsPeerAuthenticationFailureBeforeItsPeerSeesIt(t *testing.T) {
	supervisor, probe, served := initializeTestBroker(
		t,
		brokerOptions{applySignerFD: -1, evalSignerFD: -1},
		testSigningKey(0x11), testSigningKey(0x22),
	)
	// No reply: the peer learns of this failure only from EOF.
	if _, err := supervisor.Write([]byte{brokerPeerChallenge + 1}); err != nil {
		t.Fatal(err)
	}
	err := <-served
	assertReportedFirst(
		t, err, probe, supervisor,
		"signing broker: could not authenticate broker peer: broker peer challenge is invalid\n",
		0, "",
	)
}
