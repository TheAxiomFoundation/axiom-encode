//go:build linux

package main

import (
	"bytes"
	"crypto/ed25519"
	"crypto/rand"
	"encoding/base64"
	"encoding/json"
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

type rootControllerSupervisorState struct {
	Guardian   int `json:"guardian"`
	Launcher   int `json:"launcher"`
	Supervisor int `json:"supervisor"`
	Broker     int `json:"broker"`
	Model      int `json:"model"`
}

type rootControllerAttackState struct {
	Launcher          int    `json:"launcher"`
	Signer            int    `json:"signer"`
	Supervisor        int    `json:"supervisor"`
	Broker            int    `json:"broker"`
	Model             int    `json:"model"`
	Sleeper           int    `json:"sleeper"`
	LauncherStopError string `json:"launcher_stop_error"`
	LauncherKillError string `json:"launcher_kill_error"`
	SignerStopError   string `json:"signer_stop_error"`
	SignerKillError   string `json:"signer_kill_error"`
}

func TestRootControllerRevokesSignerOnVerifierTerminationAndDrainsEscapes(t *testing.T) {
	testCases := []struct {
		name             string
		signal           syscall.Signal
		expectedExitCode int
	}{
		{name: "stopped", signal: syscall.SIGSTOP, expectedExitCode: 2},
		{name: "killed", signal: syscall.SIGKILL, expectedExitCode: 137},
	}
	for _, testCase := range testCases {
		t.Run(testCase.name, func(t *testing.T) {
			testRootControllerRevokesSignerAndDrainsEscapes(
				t,
				testCase.signal,
				testCase.expectedExitCode,
			)
		})
	}
}

func testRootControllerRevokesSignerAndDrainsEscapes(
	t *testing.T,
	terminationSignal syscall.Signal,
	expectedExitCode int,
) {
	if testing.Short() {
		unavailablePrivilegedGuardianPrerequisite(
			t,
			"root process-boundary integration is disabled by -short",
		)
	}
	rootCommand := privilegedGuardianTestCommand(t)
	if os.Geteuid() != 0 {
		if output, err := rootCommand("/usr/bin/true").CombinedOutput(); err != nil {
			unavailablePrivilegedGuardianPrerequisite(
				t,
				"root execution and passwordless sudo are unavailable: %v (%s)",
				err,
				output,
			)
		}
	}

	buildDirectory := t.TempDir()
	guardianCandidate := filepath.Join(buildDirectory, "axiom-encode-process-guardian")
	applySignerCandidate := filepath.Join(buildDirectory, "axiom-encode-apply-signer")
	supervisorCandidate := filepath.Join(buildDirectory, "axiom-encode-signing-supervisor")
	buildStaticGoBinary(t, guardianCandidate, ".")
	buildStaticGoBinary(t, applySignerCandidate, "../axiom-encode-apply-signer")
	supervisorSource := filepath.Join(buildDirectory, "root_controller_adversary.go")
	if err := os.WriteFile(supervisorSource, []byte(rootControllerAdversarySource), 0o600); err != nil {
		t.Fatal(err)
	}
	buildStaticGoBinary(t, supervisorCandidate, supervisorSource)

	unique := fmt.Sprintf("%d-%d", os.Getpid(), time.Now().UnixNano())
	protectedDirectory := filepath.Join("/opt", "axiom-root-controller-test-"+unique)
	stateDirectory := filepath.Join(os.TempDir(), "axiom-root-controller-state-"+unique)
	guardian := filepath.Join(protectedDirectory, "axiom-encode-process-guardian")
	applySigner := filepath.Join(protectedDirectory, "axiom-encode-apply-signer")
	supervisor := filepath.Join(protectedDirectory, "axiom-encode-signing-supervisor")
	for _, command := range []*exec.Cmd{
		rootCommand("/usr/bin/install", "-d", "-o", "0", "-g", "0", "-m", "0755", protectedDirectory),
		rootCommand("/usr/bin/install", "-o", "0", "-g", "0", "-m", "0555", guardianCandidate, guardian),
		rootCommand("/usr/bin/install", "-o", "0", "-g", "0", "-m", "0555", applySignerCandidate, applySigner),
		rootCommand("/usr/bin/install", "-o", "0", "-g", "0", "-m", "0555", supervisorCandidate, supervisor),
	} {
		if output, err := command.CombinedOutput(); err != nil {
			t.Fatalf("could not install protected test binary: %v\n%s", err, output)
		}
	}
	if err := os.Mkdir(stateDirectory, 0o777); err != nil {
		t.Fatal(err)
	}
	if err := os.Chmod(stateDirectory, 0o777); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		_ = rootCommand("/bin/rm", "-rf", "--", protectedDirectory).Run()
		_ = rootCommand("/bin/rm", "-rf", "--", stateDirectory).Run()
	})

	_, privateKey, err := ed25519.GenerateKey(rand.Reader)
	if err != nil {
		t.Fatal(err)
	}
	key := base64.StdEncoding.EncodeToString(privateKey.Seed())
	// Numeric identities do not need passwd entries. The high, dedicated IDs
	// avoid sharing signal authority with the runner account or each other.
	const (
		signerUID     = 61001
		signerGID     = 62001
		supervisorUID = 61002
		supervisorGID = 62002
	)
	command := rootCommand(
		"/usr/bin/env",
		"GITHUB_ACTIONS=true",
		"GITHUB_REPOSITORY=TheAxiomFoundation/axiom-encode",
		"GITHUB_WORKFLOW_REF=TheAxiomFoundation/axiom-encode/.github/workflows/signing-supervisor.yml@refs/heads/main",
		"GITHUB_EVENT_NAME=workflow_dispatch",
		"GITHUB_SHA=deadbeef",
		"GITHUB_RUN_ID=12345",
		"AXIOM_ROOT_CONTROLLER_TEST_STATE="+stateDirectory,
		"AXIOM_ROOT_CONTROLLER_TEST_KEY="+key,
		guardian,
		"run-root-controller",
		"--",
		applySigner,
		"run",
		"--signer-uid", strconv.Itoa(signerUID),
		"--signer-gid", strconv.Itoa(signerGID),
		"--supervisor-uid", strconv.Itoa(supervisorUID),
		"--supervisor-gid", strconv.Itoa(supervisorGID),
		"--scope", "apply_ed25519",
		"--key-env", "AXIOM_ROOT_CONTROLLER_TEST_KEY",
		"--supervisor", supervisor,
		"--trusted-signing-roots", "/dev/null",
		"--expected-github-repository", "TheAxiomFoundation/axiom-encode",
		"--allowed-workflow-ref", "TheAxiomFoundation/axiom-encode/.github/workflows/signing-supervisor.yml@refs/heads/main",
		"--allowed-event-name", "workflow_dispatch",
		"--",
		"/opt/axiom-signing/axiom-encode", "encode", "test", "--apply",
	)
	var output bytes.Buffer
	command.Stdout = &output
	command.Stderr = &output
	if err := command.Start(); err != nil {
		t.Fatal(err)
	}

	supervisorReadyPath := filepath.Join(stateDirectory, "supervisor.json")
	if err := waitForGuardianTestFile(supervisorReadyPath, 20*time.Second); err != nil {
		_ = command.Process.Kill()
		t.Fatalf("split supervisor did not become ready: %v\n%s", err, output.String())
	}
	rawSupervisor, err := os.ReadFile(supervisorReadyPath)
	if err != nil {
		t.Fatal(err)
	}
	var supervisorState rootControllerSupervisorState
	if err := json.Unmarshal(rawSupervisor, &supervisorState); err != nil {
		t.Fatalf("malformed supervisor state: %v (%s)", err, rawSupervisor)
	}
	childrenOutput, err := rootCommand(
		supervisor,
		"inspect-children",
		strconv.Itoa(supervisorState.Launcher),
	).Output()
	if err != nil {
		t.Fatalf("could not inspect root launcher children: %v", err)
	}
	signerPID := 0
	for _, field := range strings.Fields(string(childrenOutput)) {
		pid, parseErr := strconv.Atoi(field)
		if parseErr != nil {
			t.Fatalf("invalid child PID %q: %v", field, parseErr)
		}
		if pid != supervisorState.Supervisor {
			if signerPID != 0 {
				t.Fatalf("root launcher had multiple unknown children: %s", childrenOutput)
			}
			signerPID = pid
		}
	}
	if signerPID <= 0 {
		t.Fatalf("could not find distinct signer child: %s", childrenOutput)
	}
	for label, identity := range map[string]struct {
		pid int
		uid int
		gid int
	}{
		"signer": {
			pid: signerPID,
			uid: signerUID,
			gid: signerGID,
		},
		"verifier": {
			pid: supervisorState.Supervisor,
			uid: supervisorUID,
			gid: supervisorGID,
		},
	} {
		status, statusErr := rootCommand(
			supervisor,
			"inspect-status",
			strconv.Itoa(identity.pid),
		).Output()
		if statusErr != nil {
			t.Fatalf("could not inspect %s status: %v", label, statusErr)
		}
		assertContainedChildStatus(t, label, status, identity.uid, identity.gid)
	}
	attackRequest, err := json.Marshal(map[string]int{
		"signal": int(terminationSignal),
		"signer": signerPID,
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(
		filepath.Join(stateDirectory, "attack.json"),
		attackRequest,
		0o666,
	); err != nil {
		t.Fatal(err)
	}

	attackReadyPath := filepath.Join(stateDirectory, "attack-ready.json")
	if err := waitForGuardianTestFile(attackReadyPath, 20*time.Second); err != nil {
		_ = command.Process.Kill()
		t.Fatalf("verifier adversary did not become ready: %v\n%s", err, output.String())
	}
	rawAttack, err := os.ReadFile(attackReadyPath)
	if err != nil {
		t.Fatal(err)
	}
	var attack rootControllerAttackState
	if err := json.Unmarshal(rawAttack, &attack); err != nil {
		t.Fatalf("malformed attack state: %v (%s)", err, rawAttack)
	}
	if attack.LauncherStopError != syscall.EPERM.Error() ||
		attack.LauncherKillError != syscall.EPERM.Error() ||
		attack.SignerStopError != syscall.EPERM.Error() ||
		attack.SignerKillError != syscall.EPERM.Error() {
		t.Fatalf("verifier could signal a protected identity: %#v", attack)
	}

	pinned := make(map[string]int)
	for label, pid := range map[string]int{
		"launcher":   attack.Launcher,
		"signer":     attack.Signer,
		"supervisor": attack.Supervisor,
		"broker":     attack.Broker,
		"model":      attack.Model,
		"sleeper":    attack.Sleeper,
	} {
		pidFD, openErr := unix.PidfdOpen(pid, 0)
		if openErr != nil {
			t.Fatalf("could not pin %s PID %d: %v", label, pid, openErr)
		}
		defer unix.Close(pidFD)
		pinned[label] = pidFD
	}
	if err := os.WriteFile(
		filepath.Join(stateDirectory, "continue"),
		[]byte("terminate supervisor\n"),
		0o666,
	); err != nil {
		t.Fatal(err)
	}

	done := make(chan error, 1)
	go func() { done <- command.Wait() }()
	select {
	case waitErr := <-done:
		var exitError *exec.ExitError
		if !errors.As(waitErr, &exitError) ||
			exitError.ExitCode() != expectedExitCode {
			t.Fatalf(
				"terminated supervisor did not fail the guarded run closed: %v\n%s",
				waitErr,
				output.String(),
			)
		}
	case <-time.After(20 * time.Second):
		_ = command.Process.Kill()
		t.Fatalf("root controller did not finish cleanup\n%s", output.String())
	}
	for label, pidFD := range pinned {
		if err := unix.PidfdSendSignal(pidFD, 0, nil, 0); !errors.Is(err, syscall.ESRCH) {
			t.Fatalf("%s survived or was not reaped: %v", label, err)
		}
	}
}

func assertContainedChildStatus(
	t *testing.T,
	label string,
	raw []byte,
	uid int,
	gid int,
) {
	t.Helper()
	wantUID, wantGID := strconv.Itoa(uid), strconv.Itoa(gid)
	seenUID, seenGID, seenGroups, seenNNP, capabilityLines := false, false, false, false, 0
	for _, line := range strings.Split(string(raw), "\n") {
		fields := strings.Fields(line)
		switch {
		case strings.HasPrefix(line, "Uid:"):
			seenUID = len(fields) == 5 && fields[1] == wantUID &&
				fields[2] == wantUID && fields[3] == wantUID && fields[4] == wantUID
		case strings.HasPrefix(line, "Gid:"):
			seenGID = len(fields) == 5 && fields[1] == wantGID &&
				fields[2] == wantGID && fields[3] == wantGID && fields[4] == wantGID
		case strings.HasPrefix(line, "Groups:"):
			seenGroups = len(fields) == 1
		case strings.HasPrefix(line, "NoNewPrivs:"):
			seenNNP = len(fields) == 2 && fields[1] == "1"
		case strings.HasPrefix(line, "CapInh:") ||
			strings.HasPrefix(line, "CapPrm:") ||
			strings.HasPrefix(line, "CapEff:") ||
			strings.HasPrefix(line, "CapAmb:"):
			capabilityLines++
			if len(fields) != 2 || fields[1] != "0000000000000000" {
				t.Fatalf("%s retained capability: %s", label, line)
			}
		}
	}
	if !seenUID || !seenGID || !seenGroups || !seenNNP || capabilityLines != 4 {
		t.Fatalf("%s status did not prove complete containment:\n%s", label, raw)
	}
}

func privilegedGuardianTestCommand(t *testing.T) func(...string) *exec.Cmd {
	t.Helper()
	return func(arguments ...string) *exec.Cmd {
		if os.Geteuid() == 0 {
			return exec.Command(arguments[0], arguments[1:]...)
		}
		sudo, err := exec.LookPath("sudo")
		if err != nil {
			unavailablePrivilegedGuardianPrerequisite(
				t,
				"root execution and passwordless sudo are unavailable",
			)
			return nil
		}
		prefixed := append([]string{"-n", "--"}, arguments...)
		return exec.Command(sudo, prefixed...)
	}
}

func buildStaticGoBinary(t *testing.T, output string, source string) {
	t.Helper()
	command := exec.Command("go", "build", "-o", output, source)
	command.Env = append(os.Environ(), "CGO_ENABLED=0")
	if combined, err := command.CombinedOutput(); err != nil {
		t.Fatalf("could not build static test binary %s: %v\n%s", source, err, combined)
	}
}

const rootControllerAdversarySource = `package main

import (
    "encoding/json"
    "errors"
    "fmt"
    "os"
    "os/exec"
    "path/filepath"
    "strconv"
    "strings"
    "syscall"
    "time"
)

func main() {
    if len(os.Args) >= 3 {
        pid := positiveInteger(os.Args[2])
        if os.Args[1] == "inspect-children" {
            raw, err := os.ReadFile(fmt.Sprintf("/proc/%d/task/%d/children", pid, pid))
            must(err)
            _, err = os.Stdout.Write(raw)
            must(err)
            return
        }
        if os.Args[1] == "inspect-status" {
            raw, err := os.ReadFile(fmt.Sprintf("/proc/%d/status", pid))
            must(err)
            _, err = os.Stdout.Write(raw)
            must(err)
            return
        }
    }
    if len(os.Args) >= 2 {
        switch os.Args[1] {
        case "broker": broker(); return
        case "model": model(); return
        case "detacher": detacher(); return
        case "sleeper": sleeper(); return
        }
    }
    supervisor()
}

func supervisor() {
    assertVerifierIdentity()
    state := requiredEnv("AXIOM_ROOT_CONTROLLER_TEST_STATE")
    self, err := os.Executable()
    must(err)
    brokerFD := duplicate(3)
    brokerCommand := exec.Command(self, "broker")
    brokerCommand.ExtraFiles = []*os.File{brokerFD}
    must(brokerCommand.Start())
    must(brokerFD.Close())
    modelFD := duplicate(3)
    modelCommand := exec.Command(self, "model")
    modelCommand.ExtraFiles = []*os.File{modelFD}
    must(modelCommand.Start())
    must(modelFD.Close())
    launcher := os.Getppid()
    guardian := processParent(launcher)
    writeJSON(filepath.Join(state, "supervisor.json"), map[string]int{
        "guardian": guardian,
        "launcher": launcher,
        "supervisor": os.Getpid(),
        "broker": brokerCommand.Process.Pid,
        "model": modelCommand.Process.Pid,
    })
    for { time.Sleep(time.Hour) }
}

func broker() {
    assertVerifierIdentity()
    descriptor := os.NewFile(3, "retained-signer")
    if descriptor == nil { panic("broker missing signer descriptor") }
    buffer := make([]byte, 1)
    _, _ = descriptor.Read(buffer)
    for { time.Sleep(time.Hour) }
}

func model() {
    assertVerifierIdentity()
    state := requiredEnv("AXIOM_ROOT_CONTROLLER_TEST_STATE")
    waitFile(filepath.Join(state, "attack.json"))
    raw, err := os.ReadFile(filepath.Join(state, "attack.json"))
    must(err)
    request := map[string]int{}
    must(json.Unmarshal(raw, &request))
    signer := request["signer"]
    if signer <= 0 { panic("missing signer PID") }
    terminationSignal := syscall.Signal(request["signal"])
    if terminationSignal != syscall.SIGSTOP && terminationSignal != syscall.SIGKILL {
        panic("invalid supervisor termination signal")
    }
    supervisor := os.Getppid()
    launcher := processParent(supervisor)
    launcherStop := syscall.Kill(launcher, syscall.SIGSTOP)
    launcherKill := syscall.Kill(launcher, syscall.SIGKILL)
    signerStop := syscall.Kill(signer, syscall.SIGSTOP)
    signerKill := syscall.Kill(signer, syscall.SIGKILL)
    if !errors.Is(launcherStop, syscall.EPERM) || !errors.Is(launcherKill, syscall.EPERM) ||
        !errors.Is(signerStop, syscall.EPERM) || !errors.Is(signerKill, syscall.EPERM) {
        panic(fmt.Sprintf("protected signal results: launcher-stop=%v launcher-kill=%v signer-stop=%v signer-kill=%v", launcherStop, launcherKill, signerStop, signerKill))
    }
    _, err = syscall.Setsid()
    must(err)
    self, err := os.Executable()
    must(err)
    detachedFD := duplicate(3)
    detached := exec.Command(self, "detacher")
    detached.ExtraFiles = []*os.File{detachedFD}
    must(detached.Start())
    must(detachedFD.Close())
    sleeperPath := filepath.Join(state, "sleeper.pid")
    waitFile(sleeperPath)
    rawSleeper, err := os.ReadFile(sleeperPath)
    must(err)
    sleeper := positiveInteger(strings.TrimSpace(string(rawSleeper)))
    supervisorRaw, err := os.ReadFile(filepath.Join(state, "supervisor.json"))
    must(err)
    supervisorState := map[string]int{}
    must(json.Unmarshal(supervisorRaw, &supervisorState))
    writeJSON(filepath.Join(state, "attack-ready.json"), map[string]any{
        "launcher": launcher,
        "signer": signer,
        "supervisor": supervisor,
        "broker": supervisorState["broker"],
        "model": os.Getpid(),
        "sleeper": sleeper,
        "launcher_stop_error": launcherStop.Error(),
        "launcher_kill_error": launcherKill.Error(),
        "signer_stop_error": signerStop.Error(),
        "signer_kill_error": signerKill.Error(),
    })
    waitFile(filepath.Join(state, "continue"))
    must(syscall.Kill(supervisor, terminationSignal))
    for { time.Sleep(time.Hour) }
}

func detacher() {
    assertVerifierIdentity()
    state := requiredEnv("AXIOM_ROOT_CONTROLLER_TEST_STATE")
    self, err := os.Executable()
    must(err)
    inheritedFD := duplicate(3)
    child := exec.Command(self, "sleeper")
    child.ExtraFiles = []*os.File{inheritedFD}
    must(child.Start())
    must(inheritedFD.Close())
    _ = state
}

func sleeper() {
    assertVerifierIdentity()
    state := requiredEnv("AXIOM_ROOT_CONTROLLER_TEST_STATE")
    descriptor := os.NewFile(3, "retained-signer")
    if descriptor == nil { panic("sleeper missing signer descriptor") }
    must(os.WriteFile(filepath.Join(state, "sleeper.pid"), []byte(strconv.Itoa(os.Getpid())), 0644))
    for { time.Sleep(time.Hour) }
}

func assertVerifierIdentity() {
    if os.Geteuid() == 0 { panic("verifier unexpectedly retained root") }
    raw, err := os.ReadFile("/proc/self/status")
    must(err)
    uid, gid := strconv.Itoa(os.Geteuid()), strconv.Itoa(os.Getegid())
    seenUID, seenGID, seenGroups, seenNNP, capabilityLines := false, false, false, false, 0
    for _, line := range strings.Split(string(raw), "\n") {
        fields := strings.Fields(line)
        switch {
        case strings.HasPrefix(line, "Uid:"):
            seenUID = len(fields) == 5 && fields[1] == uid && fields[2] == uid && fields[3] == uid && fields[4] == uid
        case strings.HasPrefix(line, "Gid:"):
            seenGID = len(fields) == 5 && fields[1] == gid && fields[2] == gid && fields[3] == gid && fields[4] == gid
        case strings.HasPrefix(line, "Groups:"):
            seenGroups = len(fields) == 1
        case strings.HasPrefix(line, "NoNewPrivs:"):
            seenNNP = len(fields) == 2 && fields[1] == "1"
        case strings.HasPrefix(line, "CapInh:") || strings.HasPrefix(line, "CapPrm:") || strings.HasPrefix(line, "CapEff:") || strings.HasPrefix(line, "CapAmb:"):
            capabilityLines++
            if len(fields) != 2 || fields[1] != "0000000000000000" { panic("verifier retained capability: "+line) }
        }
    }
    if !seenUID || !seenGID || !seenGroups || !seenNNP || capabilityLines != 4 {
        panic("verifier identity containment was incomplete")
    }
}

func duplicate(fd int) *os.File {
    copied, err := syscall.Dup(fd)
    must(err)
    return os.NewFile(uintptr(copied), "retained-signer")
}

func processParent(pid int) int {
    raw, err := os.ReadFile(fmt.Sprintf("/proc/%d/status", pid))
    must(err)
    for _, line := range strings.Split(string(raw), "\n") {
        if strings.HasPrefix(line, "PPid:") {
            fields := strings.Fields(line)
            if len(fields) == 2 { return positiveInteger(fields[1]) }
        }
    }
    panic("process parent unavailable")
}

func positiveInteger(raw string) int {
    value, err := strconv.Atoi(raw)
    must(err)
    if value <= 0 { panic("expected positive integer") }
    return value
}

func requiredEnv(name string) string {
    value := os.Getenv(name)
    if value == "" { panic("missing "+name) }
    return value
}

func writeJSON(path string, value any) {
    raw, err := json.Marshal(value)
    must(err)
    must(os.WriteFile(path, raw, 0644))
}

func waitFile(path string) {
    deadline := time.Now().Add(15*time.Second)
    for time.Now().Before(deadline) {
        if _, err := os.Stat(path); err == nil { return }
        time.Sleep(5*time.Millisecond)
    }
    panic("timed out waiting for "+path)
}

func must(err error) {
    if err != nil { panic(err) }
}
`
