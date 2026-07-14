//go:build darwin || linux

package main

import (
	"bytes"
	"crypto/ed25519"
	"crypto/rand"
	"encoding/base64"
	"net"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"syscall"
	"testing"
)

func socketPairFiles(t *testing.T) (*os.File, *os.File, error) {
	t.Helper()
	descriptors, err := syscall.Socketpair(syscall.AF_UNIX, syscall.SOCK_STREAM, 0)
	if err != nil {
		return nil, nil, err
	}
	// Mark both close-on-exec so only the end explicitly placed in ExtraFiles
	// reaches the signer child; otherwise the peer end leaks into the child and
	// it never observes EOF.
	syscall.CloseOnExec(descriptors[0])
	syscall.CloseOnExec(descriptors[1])
	return os.NewFile(uintptr(descriptors[0]), "signer-end"),
		os.NewFile(uintptr(descriptors[1]), "peer-end"), nil
}

var applySignerBinary string

func TestMain(m *testing.M) {
	dir, err := os.MkdirTemp("", "apply-signer-build")
	if err != nil {
		panic(err)
	}
	defer os.RemoveAll(dir)
	applySignerBinary = filepath.Join(dir, "axiom-encode-apply-signer")
	build := exec.Command("go", "build", "-o", applySignerBinary, ".")
	build.Env = append(os.Environ(), "CGO_ENABLED=0")
	if output, err := build.CombinedOutput(); err != nil {
		panic("build failed: " + string(output))
	}
	os.Exit(m.Run())
}

// stubSupervisor writes a script that records its argv and environment, so a
// launcher test can assert the key never reaches the supervised process.
func stubSupervisor(t *testing.T, dumpPath string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "stub-supervisor")
	script := "#!/bin/sh\n" +
		"{ echo \"ARGV: $*\"; echo '---ENV---'; env; } > " + shellQuote(dumpPath) + "\n" +
		"exit 0\n"
	if err := os.WriteFile(path, []byte(script), 0o755); err != nil {
		t.Fatal(err)
	}
	return path
}

func shellQuote(value string) string {
	return "'" + strings.ReplaceAll(value, "'", `'\''`) + "'"
}

func ciEnvironmentPairs() []string {
	return []string{
		"GITHUB_ACTIONS=true",
		"GITHUB_REPOSITORY=TheAxiomFoundation/rulespec-uk",
		"GITHUB_WORKFLOW_REF=TheAxiomFoundation/rulespec-uk/.github/workflows/bulk-encode.yml@refs/heads/main",
		"GITHUB_EVENT_NAME=workflow_dispatch",
		"GITHUB_SHA=deadbeef",
		"GITHUB_RUN_ID=99",
		"PATH=" + os.Getenv("PATH"),
	}
}

func TestLauncherKeepsKeyOutOfSupervisorEnvironment(t *testing.T) {
	_, private, err := ed25519.GenerateKey(rand.Reader)
	if err != nil {
		t.Fatal(err)
	}
	seedB64 := base64.StdEncoding.EncodeToString(private.Seed())
	dump := filepath.Join(t.TempDir(), "supervisor-dump.txt")
	supervisor := stubSupervisor(t, dump)

	command := exec.Command(applySignerBinary,
		"run",
		"--scope", "apply_ed25519",
		"--key-env", "APPLY_SIGNER_TEST_KEY",
		"--signer-executable", applySignerBinary,
		"--supervisor", supervisor,
		"--trusted-signing-roots", "/dev/null",
		"--expected-github-repository", "TheAxiomFoundation/rulespec-uk",
		"--allowed-workflow-ref", "TheAxiomFoundation/rulespec-uk/.github/workflows/bulk-encode.yml@refs/heads/main",
		"--allowed-event-name", "workflow_dispatch",
		"--", "/opt/axiom-signing/axiom-encode", "encode", "uk/statute/toy", "--apply",
	)
	command.Env = append(ciEnvironmentPairs(), "APPLY_SIGNER_TEST_KEY="+seedB64)
	output, err := command.CombinedOutput()
	if err != nil {
		t.Fatalf("launcher failed: %v\n%s", err, output)
	}

	dumped, err := os.ReadFile(dump)
	if err != nil {
		t.Fatalf("supervisor stub did not run: %v", err)
	}
	record := string(dumped)
	if strings.Contains(record, seedB64) {
		t.Fatal("supervisor environment/argv leaked the private key material")
	}
	if strings.Contains(record, "APPLY_SIGNER_TEST_KEY") {
		t.Fatal("supervisor inherited the key environment variable name")
	}
	if !strings.Contains(record, "--apply-signer-fd 3") {
		t.Fatalf("supervisor was not invoked with the attached signer fd: %s", record)
	}
	if !strings.Contains(record, "encode uk/statute/toy --apply") {
		t.Fatalf("supervisor did not receive the encode command: %s", record)
	}
	// The key must never appear in the launcher's own stdout/stderr either.
	if strings.Contains(string(output), seedB64) {
		t.Fatal("launcher output leaked the private key material")
	}
}

func TestLauncherRefusesOutsideActions(t *testing.T) {
	command := exec.Command(applySignerBinary,
		"run",
		"--scope", "apply_ed25519",
		"--key-env", "APPLY_SIGNER_TEST_KEY",
		"--supervisor", "/bin/true",
		"--trusted-signing-roots", "/dev/null",
		"--expected-github-repository", "TheAxiomFoundation/rulespec-uk",
		"--allowed-workflow-ref", "ref",
		"--allowed-event-name", "workflow_dispatch",
		"--", "/opt/axiom-signing/axiom-encode", "encode", "x", "--apply",
	)
	command.Env = []string{"PATH=" + os.Getenv("PATH"), "APPLY_SIGNER_TEST_KEY=x"}
	output, err := command.CombinedOutput()
	if err == nil {
		t.Fatalf("expected launcher to refuse outside Actions, got success: %s", output)
	}
	if !strings.Contains(string(output), "outside GitHub Actions") {
		t.Fatalf("expected outside-Actions refusal, got: %s", output)
	}
}

func TestServeLocalDevIgnoresKeyAndSelfGenerates(t *testing.T) {
	// A real throwaway key placed on the key-fd must be ignored in local-dev
	// mode: the signer serves with a self-generated key, so the challenge
	// response carries a different public key than the one provided.
	_, provided, err := ed25519.GenerateKey(rand.Reader)
	if err != nil {
		t.Fatal(err)
	}
	providedPublic := provided.Public().(ed25519.PublicKey)
	providedPublicB64 := base64.StdEncoding.EncodeToString(providedPublic)

	reader, writer, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	signerEnd, peerEnd, err := socketPairFiles(t)
	if err != nil {
		t.Fatal(err)
	}

	command := exec.Command(applySignerBinary,
		"serve",
		"--scope", "apply_ed25519",
		"--socket-fd", "3",
		"--key-fd", "4",
		"--allow-local-dev",
	)
	command.Env = []string{"PATH=" + os.Getenv("PATH")} // deliberately NOT in Actions
	command.ExtraFiles = []*os.File{signerEnd, reader}
	output := &bytes.Buffer{}
	command.Stdout = output
	command.Stderr = output

	if err := command.Start(); err != nil {
		t.Fatal(err)
	}
	_ = signerEnd.Close()
	_ = reader.Close()
	_, _ = writer.Write([]byte(base64.StdEncoding.EncodeToString(provided.Seed())))
	_ = writer.Close()

	// Drive the challenge handshake over the peer end, exactly as the broker
	// would. The connection stays open until the round-trip completes, so the
	// signer attaches its socket while still connected.
	conn, err := net.FileConn(peerEnd)
	if err != nil {
		t.Fatal(err)
	}
	_ = peerEnd.Close()
	broker := &brokerEmulator{t: t, conn: conn}
	nonce := make([]byte, 32)
	if _, err := rand.Read(nonce); err != nil {
		t.Fatal(err)
	}
	broker.send(map[string]any{
		"version": 2, "id": 1, "op": "challenge",
		"scope": scopeApply, "challenge": nonce,
	})
	fields, _ := broker.receive()
	servedPublic := decodeBytesField(t, fields, "public_key")
	if bytes.Equal(servedPublic, providedPublic) {
		t.Fatal("local-dev mode signed with the provided key instead of a throwaway")
	}
	challengeMessage := append([]byte(signerChallengeDomain+scopeApply+"\x00"), nonce...)
	if !ed25519.Verify(servedPublic, challengeMessage, decodeBytesField(t, fields, "signature")) {
		t.Fatal("local-dev challenge signature does not verify")
	}
	_ = conn.Close()
	_ = command.Wait()

	if strings.Contains(output.String(), providedPublicB64) {
		t.Fatalf("local-dev output leaked/used the provided key: %s", output.String())
	}
	if !strings.Contains(output.String(), "mode=local-dev") {
		t.Fatalf("expected local-dev bind audit, got: %s", output.String())
	}
}
