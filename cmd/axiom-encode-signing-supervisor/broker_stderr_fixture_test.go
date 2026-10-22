//go:build (darwin || linux) && signing_supervisor_test_fixture

package main

import (
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"testing"
)

const closedStderrBrokerEnv = "AXIOM_ENCODE_TEST_CLOSED_STDERR_BROKER"

// The fixture trust policy accepts this test binary, so the test can run the
// real broker entry point (main) as a child process whose stderr is a closed
// pipe.
func TestBrokerRepliesWhenItsStderrIsClosed(t *testing.T) {
	if os.Getenv(closedStderrBrokerEnv) == "1" {
		os.Args = []string{os.Args[0], "__broker", "--apply-signer-fd", "4"}
		main()
		os.Exit(0)
	}
	executable, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	// The broker requires a canonical path to its own executable.
	if executable, err = filepath.EvalSymlinks(executable); err != nil {
		t.Fatal(err)
	}
	applyKey := testSigningKey(0x11)
	broker, supervisor := brokerTestConnection(t)
	signer := os.NewFile(
		uintptr(startTestSigner(t, applyKey, wrongChallengeSignature)), "test-apply-signer",
	)
	stderrReader, stderrWriter, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	_ = stderrReader.Close()

	command := exec.Command(executable, "-test.run=^TestBrokerRepliesWhenItsStderrIsClosed$")
	command.Env = []string{closedStderrBrokerEnv + "=1"}
	command.ExtraFiles = []*os.File{broker, signer} // descriptors 3 and 4
	command.Stderr = stderrWriter
	startErr := command.Start()
	_ = broker.Close()
	_ = signer.Close()
	_ = stderrWriter.Close()
	if startErr != nil {
		t.Fatal(startErr)
	}

	if err := sendFrame(supervisor, testInitialization(applyKey, testSigningKey(0x22))); err != nil {
		t.Fatal(err)
	}
	var response receivedBrokerResponse
	receiveErr := receiveFrame(supervisor, &response)
	waitErr := command.Wait()
	if receiveErr != nil || response.Version != protocolVersion || response.ID != 0 ||
		response.OK || response.Error != "External apply signer initialization failed" {
		t.Fatalf(
			"broker reply = %+v, %v (broker exit: %v), want the generic initialization error",
			response, receiveErr, waitErr,
		)
	}
	var exit *exec.ExitError
	if !errors.As(waitErr, &exit) || exit.ExitCode() != 2 {
		t.Fatalf("broker exit = %v, want exit status 2", waitErr)
	}
}
