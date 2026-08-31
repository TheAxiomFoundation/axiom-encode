//go:build linux

package main

import (
	"bytes"
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

func TestParseGuardianOptionsRequiresDistinctNonRootIdentity(t *testing.T) {
	for _, arguments := range [][]string{
		{"--uid", "0", "--gid", "1", "--", "/bin/true"},
		{"--uid", "1", "--gid", "0", "--", "/bin/true"},
		{"--uid", "4294967295", "--gid", "1", "--", "/bin/true"},
		{"--uid", "1", "--gid", "4294967295", "--", "/bin/true"},
		{"--uid", "4294967296", "--gid", "1", "--", "/bin/true"},
		{"--uid", "1", "--", "/bin/true"},
	} {
		if _, err := parseGuardianOptions("run", arguments); err == nil {
			t.Fatalf("expected invalid guardian identity to fail: %q", arguments)
		}
	}

	parsed, err := parseGuardianOptions("run", []string{
		"--uid", "1234", "--gid", "5678", "--", "/opt/protected", "arg",
	})
	if err != nil {
		t.Fatal(err)
	}
	if parsed.uid != 1234 || parsed.gid != 5678 || strings.Join(parsed.command, "\x00") != "/opt/protected\x00arg" {
		t.Fatalf("unexpected parsed options: %#v", parsed)
	}
}

func TestRootControllerModeIsApplySignerOnly(t *testing.T) {
	if _, err := parseGuardianOptions(
		"run-root-controller",
		[]string{"--", "/opt/axiom-encode-signing-supervisor"},
	); err == nil {
		t.Fatal("root-controller mode accepted a non-apply-signer target")
	}
	parsed, err := parseGuardianOptions(
		"run-root-controller",
		[]string{"--", "/opt/axiom-encode-apply-signer", "run"},
	)
	if err != nil || !parsed.runRootController {
		t.Fatalf("protected apply signer root-controller mode failed: %#v %v", parsed, err)
	}
	for _, command := range [][]string{
		{"--", "/opt/axiom-encode-apply-signer", "serve"},
		{"--", "/opt/axiom-encode-apply-signer", "run", "--allow-local-dev"},
		{"--", "/opt/axiom-encode-apply-signer", "run", "--allow-local-dev=true"},
	} {
		if _, err := parseGuardianOptions("run-root-controller", command); err == nil {
			t.Fatalf("root controller accepted unsafe command: %q", command)
		}
	}
}

func TestDescendantDrainHasNoAttackerControlledDepthLimit(t *testing.T) {
	const depth = 8192
	remaining := depth
	terminated := 0
	err := drainGuardianDescendantsWith(
		func() ([]int, error) {
			if remaining == 0 {
				return nil, nil
			}
			return []int{remaining}, nil
		},
		func(pid int) error {
			if pid != remaining {
				return fmt.Errorf("unexpected simulated child %d, want %d", pid, remaining)
			}
			return nil
		},
		func(pid int) error {
			remaining--
			terminated++
			return nil
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	if terminated != depth {
		t.Fatalf("drained %d simulated descendants, want %d", terminated, depth)
	}
}

func TestDescendantDrainRetriesFailureAndReportsItOnlyAfterEmpty(t *testing.T) {
	present := true
	attempts := 0
	err := drainGuardianDescendantsWith(
		func() ([]int, error) {
			if present {
				return []int{41}, nil
			}
			return nil, nil
		},
		func(pid int) error {
			attempts++
			if attempts == 1 {
				return errors.New("transient kill failure")
			}
			return nil
		},
		func(pid int) error {
			present = false
			return nil
		},
	)
	if err == nil || !strings.Contains(err.Error(), "transient kill failure") {
		t.Fatalf("cleanup fault was not preserved after a complete drain: %v", err)
	}
	if present || attempts != 2 {
		t.Fatalf("drain returned before cleanup succeeded: present=%t attempts=%d", present, attempts)
	}
}

type guardianAdversaryPIDs struct {
	Guardian   int    `json:"guardian"`
	Supervisor int    `json:"supervisor"`
	Broker     int    `json:"broker"`
	Adversary  int    `json:"adversary"`
	Sleeper    int    `json:"sleeper"`
	KillError  string `json:"guardian_kill_error"`
}

func TestRootGuardianSurvivesSameUIDSupervisorKillAndDrainsEscapes(t *testing.T) {
	if _, err := exec.LookPath("go"); err != nil {
		unavailablePrivilegedGuardianPrerequisite(t, "Go toolchain is unavailable")
	}
	rootCommand := func(arguments ...string) *exec.Cmd {
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

	uid, gid := nobodyIdentity(t)
	buildDirectory := t.TempDir()
	guardianCandidate := filepath.Join(buildDirectory, "axiom-encode-process-guardian")
	helperCandidate := filepath.Join(buildDirectory, "guardian-adversary")
	build := exec.Command("go", "build", "-o", guardianCandidate, ".")
	if output, err := build.CombinedOutput(); err != nil {
		t.Fatalf("could not build guardian: %v\n%s", err, output)
	}
	helperSource := filepath.Join(buildDirectory, "guardian_adversary.go")
	if err := os.WriteFile(helperSource, []byte(guardianAdversarySource), 0o600); err != nil {
		t.Fatal(err)
	}
	build = exec.Command("go", "build", "-o", helperCandidate, helperSource)
	if output, err := build.CombinedOutput(); err != nil {
		t.Fatalf("could not build guardian adversary: %v\n%s", err, output)
	}

	unique := fmt.Sprintf("%d-%d", os.Getpid(), time.Now().UnixNano())
	protectedDirectory := filepath.Join("/opt", "axiom-guardian-test-"+unique)
	stateDirectory := filepath.Join(os.TempDir(), "axiom-guardian-state-"+unique)
	guardian := filepath.Join(protectedDirectory, "axiom-encode-process-guardian")
	helper := filepath.Join(protectedDirectory, "guardian-adversary")
	for _, command := range []*exec.Cmd{
		rootCommand("/usr/bin/install", "-d", "-o", "0", "-g", "0", "-m", "0755", protectedDirectory),
		rootCommand("/usr/bin/install", "-o", "0", "-g", "0", "-m", "0555", guardianCandidate, guardian),
		rootCommand("/usr/bin/install", "-o", "0", "-g", "0", "-m", "0555", helperCandidate, helper),
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
		_ = os.RemoveAll(stateDirectory)
		cleanup := rootCommand("/bin/rm", "-rf", "--", protectedDirectory)
		_ = cleanup.Run()
	})

	command := rootCommand(
		guardian,
		"run",
		"--uid", strconv.FormatUint(uint64(uid), 10),
		"--gid", strconv.FormatUint(uint64(gid), 10),
		"--",
		helper,
		"supervisor",
		stateDirectory,
	)
	var output bytes.Buffer
	command.Stdout = &output
	command.Stderr = &output
	if err := command.Start(); err != nil {
		t.Fatal(err)
	}

	readyPath := filepath.Join(stateDirectory, "ready.json")
	if err := waitForGuardianTestFile(readyPath, 15*time.Second); err != nil {
		_ = command.Process.Kill()
		t.Fatalf("adversary did not become ready: %v\n%s", err, output.String())
	}
	rawReady, err := os.ReadFile(readyPath)
	if err != nil {
		t.Fatal(err)
	}
	var pids guardianAdversaryPIDs
	if err := json.Unmarshal(rawReady, &pids); err != nil {
		t.Fatalf("malformed adversary readiness: %v (%s)", err, rawReady)
	}
	if pids.KillError != syscall.EPERM.Error() {
		t.Fatalf("verifier could signal the root guardian: %#v", pids)
	}
	pinned := make(map[string]int)
	for label, pid := range map[string]int{
		"supervisor": pids.Supervisor,
		"broker":     pids.Broker,
		"adversary":  pids.Adversary,
		"sleeper":    pids.Sleeper,
	} {
		pidFD, openErr := unix.PidfdOpen(pid, 0)
		if openErr != nil {
			t.Fatalf("could not pin %s PID %d: %v", label, pid, openErr)
		}
		defer unix.Close(pidFD)
		pinned[label] = pidFD
	}
	if err := os.WriteFile(filepath.Join(stateDirectory, "continue"), []byte("continue\n"), 0o666); err != nil {
		t.Fatal(err)
	}

	done := make(chan error, 1)
	go func() { done <- command.Wait() }()
	select {
	case waitErr := <-done:
		var exitError *exec.ExitError
		if !errors.As(waitErr, &exitError) || exitError.ExitCode() != 137 {
			t.Fatalf("guardian did not preserve the killed supervisor status: %v\n%s", waitErr, output.String())
		}
	case <-time.After(15 * time.Second):
		_ = rootCommand("/bin/kill", "-KILL", strconv.Itoa(pids.Guardian)).Run()
		t.Fatalf("guardian did not finish descendant cleanup\n%s", output.String())
	}

	for label, pidFD := range pinned {
		if err := unix.PidfdSendSignal(pidFD, 0, nil, 0); !errors.Is(err, syscall.ESRCH) {
			t.Fatalf("%s survived or was not reaped: %v", label, err)
		}
	}
}

func nobodyIdentity(t *testing.T) (uint32, uint32) {
	t.Helper()
	read := func(flag string) uint32 {
		output, err := exec.Command("id", flag, "nobody").Output()
		if err != nil {
			unavailablePrivilegedGuardianPrerequisite(
				t,
				"nobody identity is unavailable: %v",
				err,
			)
			return 0
		}
		value, err := strconv.ParseUint(strings.TrimSpace(string(output)), 10, 32)
		if err != nil || value == 0 {
			t.Fatalf("invalid nobody identity %q: %v", output, err)
		}
		return uint32(value)
	}
	return read("-u"), read("-g")
}

func waitForGuardianTestFile(path string, timeout time.Duration) error {
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		if _, err := os.Stat(path); err == nil {
			return nil
		} else if !errors.Is(err, os.ErrNotExist) {
			return err
		}
		time.Sleep(10 * time.Millisecond)
	}
	return fmt.Errorf("timed out waiting for %s", path)
}

const guardianAdversarySource = `package main

import (
    "encoding/json"
    "errors"
    "fmt"
    "os"
    "os/exec"
    "path/filepath"
    "strconv"
    "syscall"
    "time"
)

func main() {
    if len(os.Args) != 3 {
        panic("expected mode and state directory")
    }
    mode, state := os.Args[1], os.Args[2]
    self, err := os.Executable()
    must(err)
    switch mode {
    case "supervisor":
        sockets, err := syscall.Socketpair(syscall.AF_UNIX, syscall.SOCK_STREAM, 0)
        must(err)
        server := os.NewFile(uintptr(sockets[0]), "broker-server")
        client := os.NewFile(uintptr(sockets[1]), "broker-client")
        broker := exec.Command(self, "broker", state)
        broker.ExtraFiles = []*os.File{server}
        must(broker.Start())
        adversary := exec.Command(self, "adversary", state)
        adversary.ExtraFiles = []*os.File{client}
        adversary.Env = append(os.Environ(),
            "GUARDIAN_PID="+strconv.Itoa(os.Getppid()),
            "SUPERVISOR_PID="+strconv.Itoa(os.Getpid()),
            "BROKER_PID="+strconv.Itoa(broker.Process.Pid),
        )
        must(adversary.Start())
        must(server.Close())
        must(client.Close())
        for { time.Sleep(time.Hour) }
    case "broker":
        broker := os.NewFile(3, "broker")
        if broker == nil { panic("missing broker fd") }
        buffer := make([]byte, 1)
        _, _ = broker.Read(buffer)
        for { time.Sleep(time.Hour) }
    case "adversary":
        assertVerifierIdentity()
        guardian := envPID("GUARDIAN_PID")
        supervisor := envPID("SUPERVISOR_PID")
        broker := envPID("BROKER_PID")
        if !rootProcess(guardian) { panic("reported guardian is not root") }
        killErr := syscall.Kill(guardian, syscall.SIGKILL)
        if !errors.Is(killErr, syscall.EPERM) {
            panic(fmt.Sprintf("root guardian kill returned %v", killErr))
        }
        _, err := syscall.Setsid()
        must(err)
        inheritedBroker := os.NewFile(3, "broker")
        if inheritedBroker == nil { panic("missing retained broker fd") }
        detacher := exec.Command(self, "detacher", state)
        detacher.ExtraFiles = []*os.File{inheritedBroker}
        must(detacher.Start())
        sleeperPath := filepath.Join(state, "sleeper.pid")
        waitFile(sleeperPath)
        rawSleeper, err := os.ReadFile(sleeperPath)
        must(err)
        sleeper, err := strconv.Atoi(string(rawSleeper))
        must(err)
        ready, err := json.Marshal(map[string]any{
            "guardian": guardian,
            "supervisor": supervisor,
            "broker": broker,
            "adversary": os.Getpid(),
            "sleeper": sleeper,
            "guardian_kill_error": killErr.Error(),
        })
        must(err)
        must(os.WriteFile(filepath.Join(state, "ready.json"), ready, 0644))
        waitFile(filepath.Join(state, "continue"))
        must(syscall.Kill(supervisor, syscall.SIGKILL))
        for { time.Sleep(time.Hour) }
    case "detacher":
        inheritedBroker := os.NewFile(3, "broker")
        if inheritedBroker == nil { panic("missing retained broker fd") }
        sleeper := exec.Command(self, "sleeper", state)
        sleeper.ExtraFiles = []*os.File{inheritedBroker}
        must(sleeper.Start())
    case "sleeper":
        inheritedBroker := os.NewFile(3, "broker")
        if inheritedBroker == nil { panic("missing retained broker fd") }
        must(os.WriteFile(filepath.Join(state, "sleeper.pid"), []byte(strconv.Itoa(os.Getpid())), 0644))
        for { time.Sleep(time.Hour) }
    default:
        panic("unknown mode")
    }
}

func envPID(name string) int {
    pid, err := strconv.Atoi(os.Getenv(name))
    must(err)
    if pid <= 0 { panic("invalid pid") }
    return pid
}

func rootProcess(pid int) bool {
    raw, err := os.ReadFile(fmt.Sprintf("/proc/%d/status", pid))
    if err != nil { return false }
    return string(raw) != "" && contains(string(raw), "Uid:\t0\t0\t0\t0")
}

func assertVerifierIdentity() {
    if os.Geteuid() == 0 { panic("adversary unexpectedly retained root") }
    raw, err := os.ReadFile("/proc/self/status")
    must(err)
    status := string(raw)
    capabilities := 0
    groups := false
    noNewPrivileges := false
    uid := strconv.Itoa(os.Geteuid())
    gid := strconv.Itoa(os.Getegid())
    for _, line := range splitLines(status) {
        fields := fieldsOf(line)
        if contains(line, "Uid:") {
            if len(fields) != 5 || fields[1] != uid || fields[2] != uid || fields[3] != uid || fields[4] != uid {
                panic("verifier retained a mismatched user identity: "+line)
            }
        }
        if contains(line, "Gid:") {
            if len(fields) != 5 || fields[1] != gid || fields[2] != gid || fields[3] != gid || fields[4] != gid {
                panic("verifier retained a mismatched group identity: "+line)
            }
        }
        if contains(line, "CapInh:") || contains(line, "CapPrm:") || contains(line, "CapEff:") || contains(line, "CapAmb:") {
            capabilities++
            if len(fields) != 2 || fields[1] != "0000000000000000" {
                panic("verifier retained capabilities: "+line)
            }
        }
        if contains(line, "Groups:") {
            groups = true
            if len(fields) != 1 || fields[0] != "Groups:" {
                panic("verifier retained supplementary groups: "+line)
            }
        }
        if contains(line, "NoNewPrivs:") {
            noNewPrivileges = len(fields) == 2 && fields[1] == "1"
        }
    }
    if capabilities != 4 || !groups || !noNewPrivileges { panic("verifier status omitted privilege fields") }
}

func splitLines(value string) []string {
    result := []string{}
    start := 0
    for i := 0; i < len(value); i++ {
        if value[i] == '\n' {
            result = append(result, value[start:i])
            start = i+1
        }
    }
    return result
}

func fieldsOf(value string) []string {
    result := []string{}
    start := -1
    for index := 0; index <= len(value); index++ {
        space := index == len(value) || value[index] == ' ' || value[index] == '\t'
        if !space && start < 0 { start = index }
        if space && start >= 0 {
            result = append(result, value[start:index])
            start = -1
        }
    }
    return result
}

func contains(value, target string) bool {
    for i := 0; i+len(target) <= len(value); i++ {
        if value[i:i+len(target)] == target { return true }
    }
    return false
}

func waitFile(path string) {
    deadline := time.Now().Add(10*time.Second)
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
