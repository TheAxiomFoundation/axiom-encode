//go:build linux

package main

import (
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"syscall"
	"testing"
	"time"

	"golang.org/x/sys/unix"
)

const (
	containmentHelperModeEnv = "AXIOM_CONTAINMENT_HELPER_MODE"
	containmentPIDFileEnv    = "AXIOM_CONTAINMENT_PID_FILE"
	containmentAckFileEnv    = "AXIOM_CONTAINMENT_ACK_FILE"
)

func containmentHelperCommand(mode, pidFile, ackFile string) *exec.Cmd {
	command := exec.Command(
		os.Args[0],
		"-test.run=^TestContainmentAdversaryHelper$",
	)
	command.Env = append(
		os.Environ(),
		containmentHelperModeEnv+"="+mode,
		containmentPIDFileEnv+"="+pidFile,
		containmentAckFileEnv+"="+ackFile,
	)
	command.Stdout = os.Stdout
	command.Stderr = os.Stderr
	return command
}

func waitForContainmentFile(path string) error {
	deadline := time.Now().Add(10 * time.Second)
	for time.Now().Before(deadline) {
		if _, err := os.Stat(path); err == nil {
			return nil
		} else if !errors.Is(err, os.ErrNotExist) {
			return err
		}
		time.Sleep(5 * time.Millisecond)
	}
	return fmt.Errorf("timed out waiting for %s", path)
}

func TestContainmentAdversaryHelper(t *testing.T) {
	mode := os.Getenv(containmentHelperModeEnv)
	if mode == "" {
		return
	}
	pidFile := os.Getenv(containmentPIDFileEnv)
	ackFile := os.Getenv(containmentAckFileEnv)
	switch mode {
	case "background-leader":
		background := containmentHelperCommand("sleeper", pidFile, ackFile)
		if err := background.Start(); err != nil {
			t.Fatal(err)
		}
		if err := waitForContainmentFile(pidFile); err != nil {
			t.Fatal(err)
		}
		if err := waitForContainmentFile(ackFile); err != nil {
			t.Fatal(err)
		}
	case "setsid-leader":
		detacher := containmentHelperCommand("double-fork", pidFile, ackFile)
		detacher.SysProcAttr = &syscall.SysProcAttr{Setsid: true}
		if err := detacher.Start(); err != nil {
			t.Fatal(err)
		}
		if err := waitForContainmentFile(pidFile); err != nil {
			t.Fatal(err)
		}
		if err := waitForContainmentFile(ackFile); err != nil {
			t.Fatal(err)
		}
	case "double-fork":
		grandchild := containmentHelperCommand("sleeper", pidFile, ackFile)
		if err := grandchild.Start(); err != nil {
			t.Fatal(err)
		}
		if err := waitForContainmentFile(pidFile); err != nil {
			t.Fatal(err)
		}
		if err := waitForContainmentFile(ackFile); err != nil {
			t.Fatal(err)
		}
	case "sleeper":
		if err := os.WriteFile(
			pidFile,
			[]byte(strconv.Itoa(os.Getpid())),
			0600,
		); err != nil {
			t.Fatal(err)
		}
		if err := waitForContainmentFile(ackFile); err != nil {
			t.Fatal(err)
		}
		for {
			time.Sleep(time.Hour)
		}
	case "exit-23":
		os.Exit(23)
	case "no-new-privs-wrapper":
		if err := hardenProcess(); err != nil {
			t.Fatal(err)
		}
		probe := containmentHelperCommand("no-new-privs-probe", "", "")
		probe.Stdout = os.Stdout
		probe.Stderr = os.Stderr
		if err := probe.Run(); err != nil {
			t.Fatal(err)
		}
	case "no-new-privs-probe":
		rawStatus, err := os.ReadFile("/proc/self/status")
		if err != nil {
			t.Fatal(err)
		}
		if !strings.Contains(string(rawStatus), "NoNewPrivs:\t1\n") {
			t.Fatalf("child did not inherit NoNewPrivs=1:\n%s", rawStatus)
		}
		if err := exec.Command("/usr/bin/sudo", "-n", "true").Run(); err == nil {
			t.Fatal("hardened child unexpectedly elevated through passwordless sudo")
		}
	default:
		t.Fatalf("unknown containment helper mode %q", mode)
	}
}

type containmentTestResult struct {
	commandErr error
	cleanupErr error
}

func runAdversarialContainmentTest(t *testing.T, mode string) {
	t.Helper()
	if err := prepareDescendantContainment(); err != nil {
		t.Fatalf("could not enable subreaper: %v", err)
	}
	directory := t.TempDir()
	pidFile := filepath.Join(directory, "descendant.pid")
	ackFile := filepath.Join(directory, "continue")
	command := containmentHelperCommand(mode, pidFile, ackFile)
	configureContainedCommand(command)
	if err := command.Start(); err != nil {
		t.Fatal(err)
	}
	result := make(chan containmentTestResult, 1)
	go func() {
		commandErr, cleanupErr := waitContainedCommand(command, 0)
		result <- containmentTestResult{
			commandErr: commandErr,
			cleanupErr: cleanupErr,
		}
	}()
	if err := waitForContainmentFile(pidFile); err != nil {
		t.Fatal(err)
	}
	rawPID, err := os.ReadFile(pidFile)
	if err != nil {
		t.Fatal(err)
	}
	descendantPID, err := strconv.Atoi(strings.TrimSpace(string(rawPID)))
	if err != nil {
		t.Fatal(err)
	}
	pidFD, err := unix.PidfdOpen(descendantPID, 0)
	if err != nil {
		t.Fatalf("could not pin descendant PID %d: %v", descendantPID, err)
	}
	defer unix.Close(pidFD)
	if err := os.WriteFile(ackFile, []byte("continue\n"), 0600); err != nil {
		t.Fatal(err)
	}
	select {
	case completed := <-result:
		if completed.commandErr != nil {
			t.Fatalf("leader did not preserve successful exit: %v", completed.commandErr)
		}
		if completed.cleanupErr != nil {
			t.Fatalf("descendant cleanup failed: %v", completed.cleanupErr)
		}
	case <-time.After(10 * time.Second):
		t.Fatal("timed out draining adversarial descendants")
	}
	if err := unix.PidfdSendSignal(pidFD, 0, nil, 0); !errors.Is(err, syscall.ESRCH) {
		t.Fatalf("descendant %d survived or was not reaped: %v", descendantPID, err)
	}
}

func TestContainmentKillsBackgroundProcessGroupMember(t *testing.T) {
	runAdversarialContainmentTest(t, "background-leader")
}

func TestContainmentKillsSetsidDoubleFork(t *testing.T) {
	runAdversarialContainmentTest(t, "setsid-leader")
}

func TestContainmentPreservesLeaderExitStatus(t *testing.T) {
	if err := prepareDescendantContainment(); err != nil {
		t.Fatalf("could not enable subreaper: %v", err)
	}
	command := containmentHelperCommand("exit-23", "", "")
	configureContainedCommand(command)
	if err := command.Start(); err != nil {
		t.Fatal(err)
	}
	commandErr, cleanupErr := waitContainedCommand(command, 0)
	if cleanupErr != nil {
		t.Fatalf("successful containment changed child status: %v", cleanupErr)
	}
	var exitError *exec.ExitError
	if !errors.As(commandErr, &exitError) || exitError.ExitCode() != 23 {
		t.Fatalf("expected child exit 23, got %v", commandErr)
	}
}

func TestHardenedChildCannotElevateThroughPasswordlessSudo(t *testing.T) {
	if os.Geteuid() == 0 {
		t.Skip("the no-new-privileges sudo regression requires a non-root runner")
	}
	if _, err := os.Stat("/usr/bin/sudo"); err != nil {
		t.Skip("/usr/bin/sudo is unavailable")
	}
	command := containmentHelperCommand("no-new-privs-wrapper", "", "")
	if output, err := command.CombinedOutput(); err != nil {
		t.Fatalf("no-new-privileges probe failed: %v\n%s", err, output)
	}
}
