//go:build darwin || linux

package main

import (
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"syscall"
	"time"
)

// runOptions configures the launcher: the process manager that pre-opens the
// connected signer socket, streams the key to the signer over a pipe, and
// executes the trusted supervisor with the signer attached.
type runOptions struct {
	scope              string
	keyEnv             string
	supervisor         string
	trustRoots         string
	pythonRuntimeRoots []string
	pythonImportRoots  []string
	pythonPackageRoot  string
	binding            contextBinding
	command            []string
}

// forwardedContextEnvironment is the exact, minimal set of variables the signer
// child needs to validate its CI context. Nothing else — and never the key —
// crosses into the signer's environment.
var forwardedContextEnvironment = []string{
	"GITHUB_ACTIONS",
	"GITHUB_REPOSITORY",
	"GITHUB_WORKFLOW_REF",
	"GITHUB_EVENT_NAME",
	"GITHUB_SHA",
	"GITHUB_RUN_ID",
}

// runLauncher wires the signer and supervisor together and returns the
// supervisor's exit code (or an error before exec).
func runLauncher(options runOptions, environment func(string) string) (int, error) {
	// Harden first, before the key is read into this process (see runServe).
	if err := hardenProcess(); err != nil {
		return 0, fmt.Errorf("could not harden launcher process: %w", err)
	}
	if !isKnownScope(options.scope) {
		return 0, fmt.Errorf("--scope must be one of %q or %q", scopeApply, scopeEval)
	}
	// Fail fast with a clear message before touching key material. The signer
	// re-validates independently as the single source of truth.
	if _, err := options.binding.validate(environment); err != nil {
		return 0, err
	}

	// The signer is always this same trusted binary in `serve` mode: the raw
	// key pipe is never handed to an operator-chosen executable.
	signerExecutable, err := defaultSignerExecutable()
	if err != nil {
		return 0, err
	}

	keyMaterial, err := readAndClearKeyEnvironment(options.keyEnv)
	if err != nil {
		return 0, err
	}
	defer zero(keyMaterial)

	signerSocket, supervisorSocket, err := newConnectedSocketPair()
	if err != nil {
		return 0, err
	}
	defer signerSocket.Close()
	defer supervisorSocket.Close()

	keyRead, keyWrite, err := os.Pipe()
	if err != nil {
		return 0, fmt.Errorf("could not create key pipe: %w", err)
	}
	defer keyRead.Close()

	signerCommand, err := startSigner(options, signerExecutable, signerSocket, keyRead, environment)
	if err != nil {
		_ = keyWrite.Close()
		return 0, err
	}
	// The signer child now owns its copies; drop the parent's.
	_ = signerSocket.Close()
	_ = keyRead.Close()

	// Stream the key to the signer, then close so it observes EOF.
	writeErr := writeAllFile(keyWrite, keyMaterial)
	zero(keyMaterial)
	closeErr := keyWrite.Close()
	if writeErr != nil || closeErr != nil {
		_ = signerCommand.Process.Kill()
		_, _ = signerCommand.Process.Wait()
		return 0, fmt.Errorf("could not deliver key to signer: %w", errors.Join(writeErr, closeErr))
	}

	supervisorCommand, err := startSupervisor(options, supervisorSocket)
	if err != nil {
		_ = signerCommand.Process.Kill()
		_, _ = signerCommand.Process.Wait()
		return 0, err
	}
	_ = supervisorSocket.Close()

	supervisorErr := supervisorCommand.Wait()
	exitCode := 0
	if supervisorErr != nil {
		var exitError *exec.ExitError
		if errors.As(supervisorErr, &exitError) {
			exitCode = exitError.ExitCode()
		} else {
			exitCode = 1
		}
	}

	// The broker closes when the supervisor tree exits, so the signer observes
	// EOF and returns. Give it a short grace period, then force teardown.
	waitOrKill(signerCommand, 5*time.Second)
	return exitCode, nil
}

func readAndClearKeyEnvironment(name string) ([]byte, error) {
	if name == "" {
		return nil, errors.New("--key-env is required")
	}
	value, present := os.LookupEnv(name)
	if !present {
		return nil, fmt.Errorf("key environment variable %q is not set", name)
	}
	if value == "" {
		_ = os.Unsetenv(name)
		return nil, fmt.Errorf("key environment variable %q is empty", name)
	}
	material := []byte(value)
	// Remove the key from this process's environment before any child is
	// spawned, so no descendant (least of all the supervised encoder) can
	// inherit it. The Go string copy in `value` is released to the GC.
	if err := os.Unsetenv(name); err != nil {
		zero(material)
		return nil, fmt.Errorf("could not clear key environment variable %q: %w", name, err)
	}
	return material, nil
}

func startSigner(
	options runOptions,
	signerExecutable string,
	signerSocket, keyRead *os.File,
	environment func(string) string,
) (*exec.Cmd, error) {
	arguments := []string{
		"serve",
		"--scope", options.scope,
		"--socket-fd", "3",
		"--key-fd", "4",
		"--expected-github-repository", options.binding.expectedRepository,
	}
	for _, ref := range options.binding.allowedWorkflowRefs {
		arguments = append(arguments, "--allowed-workflow-ref", ref)
	}
	for _, event := range options.binding.allowedEventNames {
		arguments = append(arguments, "--allowed-event-name", event)
	}
	command := exec.Command(signerExecutable, arguments...)
	command.Env = minimalSignerEnvironment(environment)
	command.ExtraFiles = []*os.File{signerSocket, keyRead}
	command.Stdout = os.Stdout
	command.Stderr = os.Stderr
	if err := command.Start(); err != nil {
		return nil, fmt.Errorf("could not start external signer: %w", err)
	}
	return command, nil
}

func minimalSignerEnvironment(environment func(string) string) []string {
	clean := make([]string, 0, len(forwardedContextEnvironment))
	for _, name := range forwardedContextEnvironment {
		if value := environment(name); value != "" {
			clean = append(clean, name+"="+value)
		}
	}
	return clean
}

func startSupervisor(options runOptions, supervisorSocket *os.File) (*exec.Cmd, error) {
	arguments := []string{
		"--apply-signer-fd", "3",
		"--trusted-signing-roots", options.trustRoots,
	}
	for _, root := range options.pythonRuntimeRoots {
		arguments = append(arguments, "--trusted-python-runtime-root", root)
	}
	for _, root := range options.pythonImportRoots {
		arguments = append(arguments, "--trusted-python-import-root", root)
	}
	if options.pythonPackageRoot != "" {
		arguments = append(arguments, "--trusted-python-package-root", options.pythonPackageRoot)
	}
	arguments = append(arguments, "--")
	arguments = append(arguments, options.command...)

	command := exec.Command(options.supervisor, arguments...)
	// os.Environ() no longer contains the key variable (cleared above). The
	// supervisor performs its own forbidden-name rejection and per-child scrub.
	command.Env = os.Environ()
	command.ExtraFiles = []*os.File{supervisorSocket}
	command.Stdin = os.Stdin
	command.Stdout = os.Stdout
	command.Stderr = os.Stderr
	if err := command.Start(); err != nil {
		return nil, fmt.Errorf("could not start trusted signing supervisor: %w", err)
	}
	return command, nil
}

func newConnectedSocketPair() (*os.File, *os.File, error) {
	descriptors, err := syscall.Socketpair(syscall.AF_UNIX, syscall.SOCK_STREAM, 0)
	if err != nil {
		return nil, nil, fmt.Errorf("could not create connected signer socket: %w", err)
	}
	syscall.CloseOnExec(descriptors[0])
	syscall.CloseOnExec(descriptors[1])
	first := os.NewFile(uintptr(descriptors[0]), "apply-signer-signer-end")
	second := os.NewFile(uintptr(descriptors[1]), "apply-signer-supervisor-end")
	if first == nil || second == nil {
		if first != nil {
			_ = first.Close()
		}
		if second != nil {
			_ = second.Close()
		}
		return nil, nil, errors.New("could not wrap connected signer socket")
	}
	return first, second, nil
}

func writeAllFile(file *os.File, data []byte) error {
	for len(data) != 0 {
		written, err := file.Write(data)
		if err != nil {
			return err
		}
		if written == 0 {
			return errors.New("short write delivering key material")
		}
		data = data[written:]
	}
	return nil
}

func waitOrKill(command *exec.Cmd, grace time.Duration) {
	done := make(chan struct{})
	go func() {
		_, _ = command.Process.Wait()
		close(done)
	}()
	select {
	case <-done:
	case <-time.After(grace):
		_ = command.Process.Kill()
		<-done
	}
}

// defaultSignerExecutable returns this binary's own path so the launcher spawns
// the same trusted image in `serve` mode by default.
func defaultSignerExecutable() (string, error) {
	executable, err := os.Executable()
	if err != nil {
		return "", fmt.Errorf("could not resolve launcher executable: %w", err)
	}
	return filepath.Clean(executable), nil
}
