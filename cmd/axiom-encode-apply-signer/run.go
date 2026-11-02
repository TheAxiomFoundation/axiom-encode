//go:build darwin || linux

package main

import (
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
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
	// The process is hardened in main() before argument parsing, so the key that
	// is already resident in this process's inherited environment is never
	// exposed through /proc/self/environ while it is being read and cleared.
	if !isKnownScope(options.scope) {
		return 0, fmt.Errorf("--scope must be one of %q or %q", scopeApply, scopeEval)
	}
	// Fail fast with a clear message before touching key material. The signer
	// re-validates independently as the single source of truth.
	if _, err := options.binding.validate(environment); err != nil {
		return 0, err
	}

	// The signer is always this same trusted binary in `serve` mode, referenced
	// by its running inode (not a swappable on-disk path): the raw key pipe is
	// never handed to an operator-chosen or same-UID-replaced executable.
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
	defer keyWrite.Close()

	readyRead, readyWrite, err := os.Pipe()
	if err != nil {
		return 0, fmt.Errorf("could not create readiness pipe: %w", err)
	}
	defer readyRead.Close()
	defer readyWrite.Close()

	signerCommand, err := startSigner(options, signerExecutable, signerSocket, keyRead, readyWrite, environment)
	if err != nil {
		return 0, err
	}
	// The signer child now owns its copies; drop the parent's.
	_ = signerSocket.Close()
	_ = keyRead.Close()
	_ = readyWrite.Close()

	// Deliver the key ONLY after the signer signals it has hardened (denied core
	// dumps + ptrace + /proc/<pid>/fd access). If it exits without signaling —
	// e.g. a context-binding refusal — readyRead observes EOF and we abort here
	// without ever writing the key onto the pipe.
	if err := awaitSignerReady(readyRead); err != nil {
		_ = signerCommand.Process.Kill()
		_, _ = signerCommand.Process.Wait()
		return 0, err
	}

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
	signerSocket, keyRead, readyWrite *os.File,
	environment func(string) string,
) (*exec.Cmd, error) {
	arguments := []string{
		"serve",
		"--scope", options.scope,
		"--socket-fd", "3",
		"--key-fd", "4",
		"--ready-fd", "5",
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
	command.ExtraFiles = []*os.File{signerSocket, keyRead, readyWrite}
	command.Stdout = os.Stdout
	command.Stderr = os.Stderr
	if err := command.Start(); err != nil {
		return nil, fmt.Errorf("could not start external signer: %w", err)
	}
	return command, nil
}

func awaitSignerReady(readyRead *os.File) error {
	buffer := make([]byte, 1)
	if _, err := io.ReadFull(readyRead, buffer); err != nil {
		return fmt.Errorf("external signer did not signal readiness before key delivery: %w", err)
	}
	return nil
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

// defaultSignerExecutable returns a reference to this binary's own running image
// so the launcher spawns the same trusted code in `serve` mode. On Linux it uses
// /proc/self/exe, which the kernel resolves to the running inode — immune to a
// same-UID replacement of the on-disk path between resolution and exec. On other
// platforms it falls back to the resolved path (the CI target is Linux).
func defaultSignerExecutable() (string, error) {
	if runtime.GOOS == "linux" {
		return "/proc/self/exe", nil
	}
	executable, err := os.Executable()
	if err != nil {
		return "", fmt.Errorf("could not resolve launcher executable: %w", err)
	}
	return filepath.Clean(executable), nil
}
