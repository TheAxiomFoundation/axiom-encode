//go:build linux

// axiom-encode-process-guardian is the outer Linux process boundary for an
// untrusted Axiom Encode execution tree. It remains a root-owned, root-identity
// subreaper while it executes one exact protected native binary under a
// dedicated, numeric, non-root UID/GID. The different identity is essential:
// an arbitrary same-UID child can always SIGKILL an in-process supervisor.
package main

import (
	"debug/elf"
	"errors"
	"flag"
	"fmt"
	"io"
	"math"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"syscall"
	"time"

	"golang.org/x/sys/unix"
)

const (
	buildKind             = "production"
	prSetDumpable         = 4
	prSetNoNewPrivs       = 38
	prSetChildSubreaper   = 36
	maximumExecutableSize = 1 << 30
)

type guardianOptions struct {
	uid               uint32
	gid               uint32
	runRootController bool
	command           []string
}

func main() {
	if len(os.Args) == 2 && os.Args[1] == "--build-kind" {
		fmt.Println(buildKind)
		return
	}
	if len(os.Args) < 2 || (os.Args[1] != "run" && os.Args[1] != "run-root-controller") {
		fmt.Fprintln(
			os.Stderr,
			"usage: axiom-encode-process-guardian <run --uid UID --gid GID|run-root-controller> -- /absolute/protected/executable [arguments...]",
		)
		os.Exit(2)
	}
	options, err := parseGuardianOptions(os.Args[1], os.Args[2:])
	if err == nil {
		err = runGuardian(options)
	}
	if err == nil {
		return
	}
	var childExit *guardedCommandExit
	if errors.As(err, &childExit) {
		os.Exit(childExit.code)
	}
	fmt.Fprintf(os.Stderr, "process guardian: %v\n", err)
	os.Exit(2)
}

func parseGuardianOptions(mode string, arguments []string) (guardianOptions, error) {
	flags := flag.NewFlagSet("axiom-encode-process-guardian "+mode, flag.ContinueOnError)
	flags.SetOutput(io.Discard)
	uid := flags.Uint64("uid", math.MaxUint64, "numeric non-root execution UID")
	gid := flags.Uint64("gid", math.MaxUint64, "numeric non-root execution GID")
	if err := flags.Parse(arguments); err != nil {
		return guardianOptions{}, err
	}
	command := flags.Args()
	if len(command) == 0 {
		return guardianOptions{}, errors.New(
			"expected `-- /absolute/protected/executable [arguments...]`",
		)
	}
	parsed := guardianOptions{command: append([]string(nil), command...)}
	switch mode {
	case "run":
		if *uid == math.MaxUint64 || *gid == math.MaxUint64 {
			return guardianOptions{}, errors.New("--uid and --gid are required")
		}
		// 0xffffffff is the kernel's "do not change this ID" sentinel. Accepting
		// it would let the child retain the root guardian identity.
		if *uid == 0 || *gid == 0 || *uid >= math.MaxUint32 || *gid >= math.MaxUint32 {
			return guardianOptions{}, errors.New(
				"--uid and --gid must be non-zero unsigned 32-bit integers",
			)
		}
		parsed.uid, parsed.gid = uint32(*uid), uint32(*gid)
	case "run-root-controller":
		if *uid != math.MaxUint64 || *gid != math.MaxUint64 {
			return guardianOptions{}, errors.New(
				"run-root-controller does not accept --uid or --gid",
			)
		}
		if filepath.Base(command[0]) != "axiom-encode-apply-signer" ||
			len(command) < 2 || command[1] != "run" {
			return guardianOptions{}, errors.New(
				"run-root-controller serves only axiom-encode-apply-signer run",
			)
		}
		for _, argument := range command[2:] {
			if argument == "--allow-local-dev" ||
				strings.HasPrefix(argument, "--allow-local-dev=") {
				return guardianOptions{}, errors.New(
					"run-root-controller refuses --allow-local-dev",
				)
			}
		}
		parsed.runRootController = true
	default:
		return guardianOptions{}, errors.New("unsupported guardian mode")
	}
	return parsed, nil
}

type guardedCommandExit struct {
	code int
}

func (exit *guardedCommandExit) Error() string {
	return fmt.Sprintf("guarded command exited with status %d", exit.code)
}

func runGuardian(options guardianOptions) error {
	// PR_SET_NO_NEW_PRIVS is a per-thread attribute. Keep the goroutine on this
	// exact OS thread from hardening through fork/exec so the verifier child
	// necessarily inherits it.
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	if err := hardenGuardian(); err != nil {
		return fmt.Errorf("could not harden root guardian: %w", err)
	}
	self, err := os.Executable()
	if err != nil {
		return fmt.Errorf("could not resolve guardian executable: %w", err)
	}
	if _, err := validateProtectedNativeExecutable(self); err != nil {
		return fmt.Errorf("guardian executable is not protected: %w", err)
	}
	target, err := validateProtectedNativeExecutable(options.command[0])
	if err != nil {
		return fmt.Errorf("guarded executable is not protected: %w", err)
	}

	command := exec.Command(target, options.command[1:]...)
	command.Env = os.Environ()
	command.Stdin = os.Stdin
	command.Stdout = os.Stdout
	command.Stderr = os.Stderr
	processAttributes := &syscall.SysProcAttr{
		Setpgid:   true,
		Pdeathsig: syscall.SIGKILL,
	}
	if !options.runRootController {
		processAttributes.Credential = &syscall.Credential{
			Uid:    options.uid,
			Gid:    options.gid,
			Groups: []uint32{},
			// false is security-significant: Go must invoke setgroups(0, nil)
			// before setgid/setuid. NoSetGroups=true would retain the root
			// guardian's supplementary groups in the verifier process.
			NoSetGroups: false,
		}
	}
	command.SysProcAttr = processAttributes
	// Install cancellation handling before Start. Otherwise a TERM delivered in
	// the post-fork/pre-wait window could kill the root guardian and orphan the
	// newly created verifier tree.
	terminationSignals := make(chan os.Signal, 4)
	signal.Notify(
		terminationSignals,
		syscall.SIGHUP,
		syscall.SIGINT,
		syscall.SIGQUIT,
		syscall.SIGTERM,
	)
	defer signal.Stop(terminationSignals)
	// An inherited SIG_IGN/SA_NOCLDWAIT disposition would let the kernel reap
	// children automatically and invalidate waitid(WNOWAIT) PID pinning.
	signal.Reset(syscall.SIGCHLD)
	if err := command.Start(); err != nil {
		return fmt.Errorf("could not start guarded command: %w", err)
	}

	commandErr, cleanupErr := containGuardedCommand(command, terminationSignals)
	if cleanupErr != nil {
		return fmt.Errorf("guarded command cleanup failed: %w", cleanupErr)
	}
	if commandErr == nil {
		return nil
	}
	return &guardedCommandExit{code: commandExitCode(commandErr)}
}

func hardenGuardian() error {
	ruid, euid, suid := unix.Getresuid()
	rgid, egid, sgid := unix.Getresgid()
	if ruid != 0 || euid != 0 || suid != 0 || rgid != 0 || egid != 0 || sgid != 0 {
		return errors.New("guardian requires real, effective, and saved root UID/GID")
	}
	if err := requireGuardianCapabilities(); err != nil {
		return err
	}
	if err := syscall.Setrlimit(syscall.RLIMIT_CORE, &syscall.Rlimit{}); err != nil {
		return err
	}
	for _, setting := range []struct {
		name  string
		value uintptr
	}{
		{name: "PR_SET_DUMPABLE", value: prSetDumpable},
		{name: "PR_SET_NO_NEW_PRIVS", value: prSetNoNewPrivs},
		{name: "PR_SET_CHILD_SUBREAPER", value: prSetChildSubreaper},
	} {
		argument := uintptr(1)
		if setting.value == prSetDumpable {
			argument = 0
		}
		if _, _, errno := syscall.Syscall6(
			syscall.SYS_PRCTL,
			setting.value,
			argument,
			0,
			0,
			0,
			0,
		); errno != 0 {
			return fmt.Errorf("prctl(%s): %w", setting.name, errno)
		}
	}
	return nil
}

func requireGuardianCapabilities() error {
	raw, err := os.ReadFile("/proc/self/status")
	if err != nil {
		return fmt.Errorf("could not inspect guardian capabilities: %w", err)
	}
	const (
		capKill   = 5
		capSetGID = 6
		capSetUID = 7
	)
	required := uint64(1<<capKill | 1<<capSetGID | 1<<capSetUID)
	for _, line := range strings.Split(string(raw), "\n") {
		fields := strings.Fields(line)
		if len(fields) != 2 || fields[0] != "CapEff:" {
			continue
		}
		effective, parseErr := strconv.ParseUint(fields[1], 16, 64)
		if parseErr != nil {
			return fmt.Errorf("could not parse guardian CapEff: %w", parseErr)
		}
		if effective&required != required {
			return errors.New("guardian requires effective CAP_KILL, CAP_SETGID, and CAP_SETUID")
		}
		return nil
	}
	return errors.New("guardian capability status is missing CapEff")
}

func validateProtectedNativeExecutable(path string) (string, error) {
	if path == "" || !filepath.IsAbs(path) || filepath.Clean(path) != path {
		return "", errors.New("path must be canonical and absolute")
	}
	realPath, err := filepath.EvalSymlinks(path)
	if err != nil {
		return "", err
	}
	if realPath != path {
		return "", errors.New("path and every ancestor must be symlink-free")
	}

	current := string(filepath.Separator)
	if err := validateProtectedPathComponent(current, true); err != nil {
		return "", err
	}
	components := strings.Split(strings.TrimPrefix(path, current), current)
	for index, component := range components {
		if component == "" {
			return "", errors.New("path contains an empty component")
		}
		current = filepath.Join(current, component)
		isTarget := index == len(components)-1
		if err := validateProtectedPathComponent(current, !isTarget); err != nil {
			return "", err
		}
		info, statErr := os.Lstat(current)
		if statErr != nil {
			return "", statErr
		}
		if !isTarget {
			continue
		}
		if !info.Mode().IsRegular() || info.Mode()&(os.ModeSymlink|os.ModeSetuid|os.ModeSetgid|os.ModeSticky) != 0 {
			return "", errors.New("trusted executable must be a regular file without special mode bits")
		}
		if info.Mode().Perm()&0o555 != 0o555 {
			return "", errors.New("trusted executable must be readable and executable by the verifier")
		}
		if info.Size() <= 0 || info.Size() > maximumExecutableSize {
			return "", errors.New("trusted executable has an invalid size")
		}
	}

	file, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer file.Close()
	header := make([]byte, 4)
	if _, err := io.ReadFull(file, header); err != nil {
		return "", err
	}
	if string(header) != "\x7fELF" {
		return "", errors.New("trusted executable must be a native ELF binary")
	}
	binary, err := elf.NewFile(file)
	if err != nil {
		return "", errors.New("trusted executable must be a native ELF binary")
	}
	defer binary.Close()
	for _, program := range binary.Progs {
		if program.Type == elf.PT_INTERP {
			return "", errors.New(
				"trusted executable must be static and have no PT_INTERP loader",
			)
		}
	}
	return path, nil
}

func validateProtectedPathComponent(path string, requireDirectory bool) error {
	info, err := os.Lstat(path)
	if err != nil {
		return err
	}
	stat, ok := info.Sys().(*syscall.Stat_t)
	if !ok {
		return fmt.Errorf("could not inspect ownership of %s", path)
	}
	if stat.Uid != 0 || info.Mode().Perm()&0o022 != 0 {
		return fmt.Errorf("trusted path is not root-owned and protected: %s", path)
	}
	if requireDirectory && (!info.IsDir() || info.Mode()&os.ModeSymlink != 0) {
		return fmt.Errorf("trusted ancestor is not a real directory: %s", path)
	}
	size, capabilityErr := unix.Getxattr(path, "security.capability", nil)
	if capabilityErr == nil {
		if size != 0 {
			return fmt.Errorf("trusted path carries security.capability: %s", path)
		}
	} else if !errors.Is(capabilityErr, syscall.ENODATA) && !errors.Is(capabilityErr, syscall.ENOTSUP) {
		return fmt.Errorf("could not inspect security.capability on %s: %w", path, capabilityErr)
	}
	return nil
}

type observedExit struct {
	err error
}

func containGuardedCommand(
	command *exec.Cmd,
	terminationSignals <-chan os.Signal,
) (error, error) {
	if command.Process == nil {
		return nil, errors.New("guarded command was not started")
	}
	leaderPID := command.Process.Pid
	var cleanupErrors []error
	groupVerified := false
	if processGroup, err := syscall.Getpgid(leaderPID); err != nil {
		cleanupErrors = append(cleanupErrors, fmt.Errorf("could not verify guarded process group: %w", err))
	} else if processGroup != leaderPID {
		cleanupErrors = append(
			cleanupErrors,
			fmt.Errorf("guarded process group is %d, expected %d", processGroup, leaderPID),
		)
	} else {
		groupVerified = true
	}

	exitObserved := make(chan observedExit, 1)
	go func() {
		var processInfo unix.Siginfo
		for {
			err := unix.Waitid(
				unix.P_PID,
				leaderPID,
				&processInfo,
				unix.WEXITED|unix.WNOWAIT,
				nil,
			)
			if errors.Is(err, syscall.EINTR) {
				continue
			}
			exitObserved <- observedExit{err: err}
			return
		}
	}()

	var observation observedExit
	select {
	case observation = <-exitObserved:
	case received := <-terminationSignals:
		cleanupErrors = append(cleanupErrors, fmt.Errorf("guardian interrupted by %s", received))
		if groupVerified {
			if err := syscall.Kill(-leaderPID, syscall.SIGKILL); err != nil && !errors.Is(err, syscall.ESRCH) {
				cleanupErrors = append(cleanupErrors, fmt.Errorf("could not interrupt guarded process group: %w", err))
			}
		} else if err := command.Process.Kill(); err != nil && !errors.Is(err, os.ErrProcessDone) {
			cleanupErrors = append(cleanupErrors, fmt.Errorf("could not interrupt guarded command: %w", err))
		}
		observation = <-exitObserved
	}
	if observation.err != nil {
		cleanupErrors = append(
			cleanupErrors,
			fmt.Errorf("could not observe guarded command exit without reaping: %w", observation.err),
		)
	}

	// The exited leader remains a zombie until command.Wait below. Its PID pins
	// the negative-PID process-group identity, preventing a recycled PID from
	// redirecting SIGKILL at an unrelated group.
	if groupVerified && observation.err == nil {
		if err := syscall.Kill(-leaderPID, syscall.SIGKILL); err != nil && !errors.Is(err, syscall.ESRCH) {
			cleanupErrors = append(cleanupErrors, fmt.Errorf("could not terminate guarded process group: %w", err))
		}
	}

	waitErr := command.Wait()
	var commandErr error
	if waitErr != nil {
		var exitError *exec.ExitError
		if errors.As(waitErr, &exitError) {
			commandErr = waitErr
		} else {
			cleanupErrors = append(cleanupErrors, fmt.Errorf("could not reap guarded command: %w", waitErr))
		}
	}
	if err := drainGuardianDescendants(); err != nil {
		cleanupErrors = append(cleanupErrors, err)
	}
	// A cancellation delivered while descendants were being drained is still a
	// failed guarded operation even though cleanup ultimately reached empty.
	select {
	case received := <-terminationSignals:
		cleanupErrors = append(cleanupErrors, fmt.Errorf("guardian interrupted by %s", received))
	default:
	}
	return commandErr, errors.Join(cleanupErrors...)
}

func commandExitCode(err error) int {
	var exitError *exec.ExitError
	if !errors.As(err, &exitError) {
		return 1
	}
	status, ok := exitError.ProcessState.Sys().(syscall.WaitStatus)
	if !ok {
		return 1
	}
	if status.Exited() {
		return status.ExitStatus()
	}
	if status.Signaled() {
		return 128 + int(status.Signal())
	}
	return 1
}

func drainGuardianDescendants() error {
	// Each pass kills and reaps every current direct child. When an adopted child
	// exits, its own children become direct children of this subreaper, so the
	// repetition also closes setsid and arbitrarily deep double-fork escapes. A
	// fixed pass limit is unsafe: an attacker can construct a deeper chain while
	// retaining the signing descriptor. The guardian therefore cannot return
	// until the kernel reports that it has no children at all.
	return drainGuardianDescendantsWith(
		directGuardianChildren,
		killGuardianChild,
		reapGuardianChild,
	)
}

func drainGuardianDescendantsWith(
	childrenOfGuardian func() ([]int, error),
	killChild func(int) error,
	reapChild func(int) error,
) error {
	var firstCleanupError error
	for {
		children, err := childrenOfGuardian()
		if err != nil {
			if firstCleanupError == nil {
				firstCleanupError = err
			}
			// A transient /proc race or resource error must never turn into a
			// successful escape. Stay resident and retry; a persistent kernel
			// failure intentionally fails closed by keeping the workflow blocked.
			time.Sleep(10 * time.Millisecond)
			continue
		}
		if len(children) == 0 {
			return firstCleanupError
		}
		killed := make([]int, 0, len(children))
		roundFault := false
		// Signal the whole snapshot before blocking in any wait4. That closes the
		// fork window for every currently adopted adversary at once.
		for _, pid := range children {
			if err := killChild(pid); err != nil {
				roundFault = true
				if firstCleanupError == nil {
					firstCleanupError = err
				}
				continue
			}
			killed = append(killed, pid)
		}
		for _, pid := range killed {
			if err := reapChild(pid); err != nil {
				roundFault = true
				if firstCleanupError == nil {
					firstCleanupError = err
				}
			}
		}
		if roundFault {
			time.Sleep(10 * time.Millisecond)
		}
	}
}

func directGuardianChildren() ([]int, error) {
	// Do not census global /proc/<pid>/status: hidepid=2 can omit a live
	// other-UID verifier child and make a global scan falsely look empty. The
	// kernel's per-thread children files are self-owned and enumerate this
	// subreaper's exact direct children even on a restricted procfs mount.
	tasks, err := os.ReadDir("/proc/self/task")
	if err != nil {
		return nil, fmt.Errorf("could not enumerate guardian tasks: %w", err)
	}
	children := make(map[int]struct{})
	for _, task := range tasks {
		tid, parseErr := strconv.Atoi(task.Name())
		if parseErr != nil || tid <= 0 {
			continue
		}
		raw, readErr := os.ReadFile(
			filepath.Join("/proc/self/task", task.Name(), "children"),
		)
		if errors.Is(readErr, os.ErrNotExist) {
			// A Go runtime worker can exit between task enumeration and read.
			continue
		}
		if readErr != nil {
			return nil, fmt.Errorf("could not inspect guardian task %d children: %w", tid, readErr)
		}
		for _, field := range strings.Fields(string(raw)) {
			pid, childParseErr := strconv.Atoi(field)
			if childParseErr != nil || pid <= 0 {
				return nil, fmt.Errorf(
					"kernel returned invalid child PID %q for guardian task %d",
					field,
					tid,
				)
			}
			children[pid] = struct{}{}
		}
	}
	result := make([]int, 0, len(children))
	for pid := range children {
		result = append(result, pid)
	}
	sort.Ints(result)
	return result, nil
}

func killGuardianChild(pid int) error {
	// A direct child retains its PID until this process reaps it. Even if it
	// exits here, the zombie pins the numeric identity before this root guardian
	// signals it.
	if err := syscall.Kill(pid, syscall.SIGKILL); err != nil && !errors.Is(err, syscall.ESRCH) {
		return fmt.Errorf("could not terminate adopted child %d: %w", pid, err)
	}
	return nil
}

func reapGuardianChild(pid int) error {
	var status syscall.WaitStatus
	for {
		waitedPID, err := syscall.Wait4(pid, &status, syscall.WALL, nil)
		if errors.Is(err, syscall.EINTR) {
			continue
		}
		if errors.Is(err, syscall.ECHILD) {
			return nil
		}
		if err != nil {
			return fmt.Errorf("could not reap adopted child %d: %w", pid, err)
		}
		if waitedPID != pid {
			return fmt.Errorf("wait4 returned unexpected child PID %d for %d", waitedPID, pid)
		}
		return nil
	}
}
