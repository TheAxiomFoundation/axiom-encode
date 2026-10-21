//go:build darwin || linux

package main

import (
	"bytes"
	"crypto/ed25519"
	"errors"
	"io"
	"os"
	"strings"
	"testing"

	"golang.org/x/sys/unix"
)

// peerProbe is the broker's diagnostics writer in these tests. At every write
// it records what the supervisor end of the broker connection could already
// read: "nothing", "data" (a reply) or "eof". The supervisor kills the broker
// as soon as it sees a failure, so a diagnostic is only safe when it is
// written while the peer still sees nothing.
type peerProbe struct {
	t        *testing.T
	peer     int
	text     strings.Builder
	observed []string
}

func (probe *peerProbe) Write(p []byte) (int, error) {
	probe.observed = append(probe.observed, pendingOnPeer(probe.t, probe.peer))
	probe.text.Write(p)
	return len(p), nil
}

func pendingOnPeer(t *testing.T, descriptor int) string {
	buffer := make([]byte, 1)
	n, _, err := unix.Recvfrom(descriptor, buffer, unix.MSG_PEEK|unix.MSG_DONTWAIT)
	switch {
	case errors.Is(err, unix.EAGAIN):
		return "nothing"
	case err != nil:
		t.Errorf("could not inspect the broker's peer: %v", err)
		return "error"
	case n == 0:
		return "eof"
	default:
		return "data"
	}
}

func testSigningKey(seed byte) ed25519.PrivateKey {
	return ed25519.NewKeyFromSeed(bytes.Repeat([]byte{seed}, ed25519.SeedSize))
}

func testInitialization(applyKey, evalKey ed25519.PrivateKey) brokerRequest {
	corpusRelease := testSigningKey(0x33).Public().(ed25519.PublicKey)
	return brokerRequest{
		Version:                 protocolVersion,
		ID:                      0,
		Operation:               "initialize",
		ApplyPublicKey:          applyKey.Public().(ed25519.PublicKey),
		EvalPublicKey:           evalKey.Public().(ed25519.PublicKey),
		CorpusReleasePublicKey:  corpusRelease,
		CorpusReleasePublicKeys: [][]byte{corpusRelease},
	}
}

type testSignerBehavior int

const (
	wrongChallengeSignature testSignerBehavior = iota
	wrongSignSignature
)

// startTestSigner serves the external signer protocol on a socketpair and
// returns the descriptor the broker attaches to; the broker owns and closes it.
func startTestSigner(t *testing.T, key ed25519.PrivateKey, behavior testSignerBehavior) int {
	t.Helper()
	descriptors, err := unix.Socketpair(unix.AF_UNIX, unix.SOCK_STREAM, 0)
	if err != nil {
		t.Fatal(err)
	}
	signer := os.NewFile(uintptr(descriptors[0]), "test-external-signer")
	done := make(chan struct{})
	go func() {
		defer close(done)
		for {
			var request signerRequest
			if err := receiveFrame(signer, &request); err != nil {
				return
			}
			response := map[string]any{
				"version": signerProtocolVersion,
				"id":      request.ID,
				"ok":      true,
			}
			switch request.Operation {
			case "challenge":
				message := append([]byte(signerChallengeDomain+request.Scope+"\x00"), request.Challenge...)
				if behavior == wrongChallengeSignature {
					message = []byte("wrong")
				}
				response["public_key"] = key.Public().(ed25519.PublicKey)
				response["signature"] = ed25519.Sign(key, message)
			case "sign":
				message := append([]byte(signerSignatureDomain+request.Scope+"\x00"), request.Payload...)
				if behavior == wrongSignSignature {
					message = request.Payload
				}
				response["signature"] = ed25519.Sign(key, message)
			}
			if err := sendFrame(signer, response); err != nil {
				return
			}
		}
	}()
	t.Cleanup(func() {
		_ = signer.Close()
		<-done
	})
	return descriptors[1]
}

func brokerTestConnection(t *testing.T) (*os.File, *os.File) {
	t.Helper()
	descriptors, err := unix.Socketpair(unix.AF_UNIX, unix.SOCK_STREAM, 0)
	if err != nil {
		t.Fatal(err)
	}
	broker := os.NewFile(uintptr(descriptors[0]), "test-broker")
	supervisor := os.NewFile(uintptr(descriptors[1]), "test-supervisor")
	t.Cleanup(func() { _ = supervisor.Close() })
	return broker, supervisor
}

// assertReportedFirst checks that the broker wrote exactly wantDiagnostic while
// its peer could not yet see the failure, that the peer then receives only the
// generic wantReply (or no reply), and that the failure is not reported twice.
func assertReportedFirst(
	t *testing.T,
	err error,
	probe *peerProbe,
	supervisor *os.File,
	wantDiagnostic string,
	wantReplyID int64,
	wantReply string,
) {
	t.Helper()
	if err == nil {
		t.Fatal("broker unexpectedly succeeded")
	}
	if got := probe.text.String(); !strings.HasPrefix(got, wantDiagnostic) ||
		strings.Count(got, "\n") != 1 {
		t.Fatalf("broker diagnostic = %q, want one line starting %q", got, wantDiagnostic)
	}
	if len(probe.observed) != 1 || probe.observed[0] != "nothing" {
		t.Fatalf(
			"broker wrote its diagnostic after its peer could see the failure (peer had %v)",
			probe.observed,
		)
	}
	var again bytes.Buffer
	_ = reportBrokerFailure(&again, err)
	if again.Len() != 0 {
		t.Fatalf("broker failure would be reported twice: %q", again.String())
	}
	if wantReply != "" {
		var response receivedBrokerResponse
		if err := receiveFrame(supervisor, &response); err != nil {
			t.Fatalf("could not read the broker's reply: %v", err)
		}
		if response.Version != protocolVersion || response.ID != wantReplyID ||
			response.OK || response.Error != wantReply {
			t.Fatalf("broker reply = %+v, want generic error %q for id %d", response, wantReply, wantReplyID)
		}
	}
	if _, err := supervisor.Read(make([]byte, 1)); !errors.Is(err, io.EOF) {
		t.Fatalf("broker connection was not closed after the failure: %v", err)
	}
}

func TestBrokerReportsInitializationFailureBeforeItsPeerSeesIt(t *testing.T) {
	applyKey := testSigningKey(0x11)
	evalKey := testSigningKey(0x22)
	valid := testInitialization(applyKey, evalKey)
	malformed := valid
	malformed.ID = 1
	testCases := []struct {
		name           string
		options        func(*testing.T) brokerOptions
		initialization any
		wantDiagnostic string
		wantReply      string
	}{
		{
			name: "apply signer challenge response is invalid",
			options: func(t *testing.T) brokerOptions {
				return brokerOptions{
					applySignerFD: startTestSigner(t, applyKey, wrongChallengeSignature),
					evalSignerFD:  -1,
				}
			},
			initialization: valid,
			wantDiagnostic: "signing broker: external apply signer challenge response is invalid\n",
			wantReply:      "External apply signer initialization failed",
		},
		{
			name: "eval signer challenge response is invalid",
			options: func(t *testing.T) brokerOptions {
				return brokerOptions{
					applySignerFD: -1,
					evalSignerFD:  startTestSigner(t, evalKey, wrongChallengeSignature),
				}
			},
			initialization: valid,
			wantDiagnostic: "signing broker: external eval signer challenge response is invalid\n",
			wantReply:      "External eval signer initialization failed",
		},
		{
			name: "initialization is malformed",
			options: func(*testing.T) brokerOptions {
				return brokerOptions{applySignerFD: -1, evalSignerFD: -1}
			},
			initialization: malformed,
			wantDiagnostic: "signing broker: initialization is malformed\n",
			wantReply:      "Signing broker initialization is malformed",
		},
		{
			// No reply: the peer learns of this failure only from EOF.
			name: "initialization frame is undecodable",
			options: func(*testing.T) brokerOptions {
				return brokerOptions{applySignerFD: -1, evalSignerFD: -1}
			},
			initialization: map[string]any{
				"version": protocolVersion, "id": 0, "op": "initialize", "unexpected": true,
			},
			wantDiagnostic: `signing broker: signing broker frame is malformed: json: unknown field "unexpected"`,
		},
	}
	for _, testCase := range testCases {
		t.Run(testCase.name, func(t *testing.T) {
			options := testCase.options(t)
			broker, supervisor := brokerTestConnection(t)
			if err := sendFrame(supervisor, testCase.initialization); err != nil {
				t.Fatal(err)
			}
			probe := &peerProbe{t: t, peer: int(supervisor.Fd())}
			err := serveBrokerConnection(broker, options, probe)
			assertReportedFirst(
				t, err, probe, supervisor, testCase.wantDiagnostic, 0, testCase.wantReply,
			)
		})
	}
}

// initializeTestBroker serves the broker protocol in the background and reads
// its successful initialization reply. The broker then waits for peer
// authentication (on Linux) and requests; its result arrives on served.
func initializeTestBroker(
	t *testing.T,
	options brokerOptions,
	applyKey, evalKey ed25519.PrivateKey,
) (supervisor *os.File, probe *peerProbe, served <-chan error) {
	t.Helper()
	broker, supervisor := brokerTestConnection(t)
	probe = &peerProbe{t: t, peer: int(supervisor.Fd())}
	result := make(chan error, 1)
	go func() { result <- serveBrokerConnection(broker, options, probe) }()

	if err := sendFrame(supervisor, testInitialization(applyKey, evalKey)); err != nil {
		t.Fatal(err)
	}
	var status receivedBrokerResponse
	if err := receiveFrame(supervisor, &status); err != nil || !status.OK {
		t.Fatalf("broker initialization failed: %+v, %v", status, err)
	}
	return supervisor, probe, result
}

func TestBrokerReportsUndecodableRequestBeforeItsPeerSeesIt(t *testing.T) {
	supervisor, probe, served := initializeTestBroker(
		t,
		brokerOptions{applySignerFD: -1, evalSignerFD: -1},
		testSigningKey(0x11), testSigningKey(0x22),
	)
	completeTestBrokerPeerAuthentication(t, supervisor)
	// No reply: the peer learns of this failure only from EOF.
	if err := sendFrame(supervisor, map[string]any{
		"version": protocolVersion, "id": 1, "op": "status", "unexpected": true,
	}); err != nil {
		t.Fatal(err)
	}
	err := <-served
	assertReportedFirst(
		t, err, probe, supervisor,
		`signing broker: signing broker frame is malformed: json: unknown field "unexpected"`,
		0, "",
	)
}

func TestBrokerReportsSigningFailureBeforeItsPeerSeesIt(t *testing.T) {
	for _, label := range []string{"apply", "eval"} {
		t.Run(label, func(t *testing.T) {
			applyKey := testSigningKey(0x11)
			evalKey := testSigningKey(0x22)
			options := brokerOptions{applySignerFD: -1, evalSignerFD: -1}
			if label == "apply" {
				options.applySignerFD = startTestSigner(t, applyKey, wrongSignSignature)
			} else {
				options.evalSignerFD = startTestSigner(t, evalKey, wrongSignSignature)
			}
			supervisor, probe, served := initializeTestBroker(t, options, applyKey, evalKey)
			completeTestBrokerPeerAuthentication(t, supervisor)
			// Leave the reply unread until the broker returns, so a reply sent
			// before the diagnostic is still pending when the probe looks.
			if err := sendFrame(supervisor, map[string]any{
				"version": protocolVersion,
				"id":      1,
				"op":      label + "_ed25519_sign",
				"payload": []byte("payload"),
			}); err != nil {
				t.Fatal(err)
			}
			err := <-served
			assertReportedFirst(
				t, err, probe, supervisor,
				"signing broker: external "+label+" signer returned an invalid signature\n",
				1, "External "+label+" signer failed",
			)
		})
	}
}
