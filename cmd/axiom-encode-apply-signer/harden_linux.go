//go:build linux

package main

import (
	"debug/elf"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"syscall"

	"golang.org/x/sys/unix"
)

const (
	prSetDumpable               = 4
	prSetNoNewPrivs             = 38
	maxSupervisorExecutableSize = 1 << 30
)

// hardenProcess denies core dumps, /proc/<pid>/mem inspection, and privilege
// escalation for the process holding (or briefly handling) private key material.
func hardenProcess() error {
	if err := syscall.Setrlimit(syscall.RLIMIT_CORE, &syscall.Rlimit{Cur: 0, Max: 0}); err != nil {
		return err
	}
	if _, _, errno := syscall.Syscall6(
		syscall.SYS_PRCTL, uintptr(prSetDumpable), 0, 0, 0, 0, 0,
	); errno != 0 {
		return fmt.Errorf("prctl(PR_SET_DUMPABLE): %w", errno)
	}
	if _, _, errno := syscall.Syscall6(
		syscall.SYS_PRCTL, uintptr(prSetNoNewPrivs), 1, 0, 0, 0, 0,
	); errno != 0 {
		return fmt.Errorf("prctl(PR_SET_NO_NEW_PRIVS): %w", errno)
	}
	return nil
}

func validateLauncherSecurity(options runOptions) error {
	ruid, euid, suid := unix.Getresuid()
	rgid, egid, sgid := unix.Getresgid()
	if options.allowLocalDev {
		if ruid == 0 || euid == 0 || suid == 0 || rgid == 0 || egid == 0 || sgid == 0 {
			return errors.New("--allow-local-dev is refused for a root launcher")
		}
		return nil
	}
	if ruid != 0 || euid != 0 || suid != 0 || rgid != 0 || egid != 0 || sgid != 0 {
		return errors.New(
			"production run requires real, effective, and saved root UID/GID",
		)
	}
	if options.signerUID == 0 || options.signerGID == 0 ||
		options.supervisorUID == 0 || options.supervisorGID == 0 ||
		options.signerUID == options.supervisorUID ||
		options.signerGID == options.supervisorGID {
		return errors.New(
			"production run requires explicit distinct non-root signer and supervisor identities",
		)
	}
	return requireLauncherCapabilities()
}

func validateSupervisorExecutable(options runOptions) (string, error) {
	if options.allowLocalDev {
		return options.supervisor, nil
	}
	path := options.supervisor
	if path == "" || !filepath.IsAbs(path) || filepath.Clean(path) != path {
		return "", errors.New("supervisor path must be canonical and absolute")
	}
	if filepath.Base(path) != "axiom-encode-signing-supervisor" {
		return "", errors.New(
			"production supervisor must be named axiom-encode-signing-supervisor",
		)
	}
	realPath, err := filepath.EvalSymlinks(path)
	if err != nil {
		return "", err
	}
	if realPath != path {
		return "", errors.New("supervisor path and every ancestor must be symlink-free")
	}

	current := string(filepath.Separator)
	if err := validateProtectedSupervisorPath(current, true); err != nil {
		return "", err
	}
	components := strings.Split(strings.TrimPrefix(path, current), current)
	for index, component := range components {
		if component == "" {
			return "", errors.New("supervisor path contains an empty component")
		}
		current = filepath.Join(current, component)
		isTarget := index == len(components)-1
		if err := validateProtectedSupervisorPath(current, !isTarget); err != nil {
			return "", err
		}
		if !isTarget {
			continue
		}
		info, statErr := os.Lstat(current)
		if statErr != nil {
			return "", statErr
		}
		if !info.Mode().IsRegular() ||
			info.Mode()&(os.ModeSymlink|os.ModeSetuid|os.ModeSetgid|os.ModeSticky) != 0 {
			return "", errors.New(
				"trusted supervisor must be a regular file without special mode bits",
			)
		}
		if info.Mode().Perm()&0o555 != 0o555 || info.Mode().Perm()&0o022 != 0 {
			return "", errors.New(
				"trusted supervisor must be readable/executable and not group/world writable",
			)
		}
		if info.Size() <= 0 || info.Size() > maxSupervisorExecutableSize {
			return "", errors.New("trusted supervisor has an invalid size")
		}
	}

	file, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer file.Close()
	binary, err := elf.NewFile(file)
	if err != nil {
		return "", errors.New("trusted supervisor must be a native ELF binary")
	}
	defer binary.Close()
	for _, program := range binary.Progs {
		if program.Type == elf.PT_INTERP {
			return "", errors.New(
				"trusted supervisor must be a static ELF binary without PT_INTERP",
			)
		}
	}
	return path, nil
}

func validateProtectedSupervisorPath(path string, requireDirectory bool) error {
	info, err := os.Lstat(path)
	if err != nil {
		return err
	}
	stat, ok := info.Sys().(*syscall.Stat_t)
	if !ok {
		return fmt.Errorf("could not inspect ownership of %s", path)
	}
	if stat.Uid != 0 || info.Mode().Perm()&0o022 != 0 {
		return fmt.Errorf("trusted supervisor path is not root-owned and protected: %s", path)
	}
	if requireDirectory && (!info.IsDir() || info.Mode()&os.ModeSymlink != 0) {
		return fmt.Errorf("trusted supervisor ancestor is not a real directory: %s", path)
	}
	size, capabilityErr := unix.Getxattr(path, "security.capability", nil)
	if capabilityErr == nil {
		if size != 0 {
			return fmt.Errorf("trusted supervisor path carries security.capability: %s", path)
		}
	} else if !errors.Is(capabilityErr, syscall.ENODATA) &&
		!errors.Is(capabilityErr, syscall.ENOTSUP) {
		return fmt.Errorf(
			"could not inspect security.capability on %s: %w", path, capabilityErr,
		)
	}
	return nil
}

func requireLauncherCapabilities() error {
	raw, err := os.ReadFile("/proc/self/status")
	if err != nil {
		return fmt.Errorf("could not inspect launcher capabilities: %w", err)
	}
	const (
		capKill   = 5
		capSetGID = 6
		capSetUID = 7
	)
	required := uint64(1<<capKill | 1<<capSetGID | 1<<capSetUID)
	seenEffective := false
	seenInheritable := false
	seenAmbient := false
	for _, line := range strings.Split(string(raw), "\n") {
		fields := strings.Fields(line)
		if len(fields) != 2 {
			continue
		}
		switch fields[0] {
		case "CapEff:":
			effective, parseErr := strconv.ParseUint(fields[1], 16, 64)
			if parseErr != nil {
				return fmt.Errorf("could not parse launcher CapEff: %w", parseErr)
			}
			if effective&required != required {
				return errors.New(
					"production launcher requires effective CAP_KILL, CAP_SETGID, and CAP_SETUID",
				)
			}
			seenEffective = true
		case "CapInh:":
			if fields[1] != "0000000000000000" {
				return errors.New("production launcher must have an empty inheritable capability set")
			}
			seenInheritable = true
		case "CapAmb:":
			if fields[1] != "0000000000000000" {
				return errors.New("production launcher must have an empty ambient capability set")
			}
			seenAmbient = true
		}
	}
	if !seenEffective || !seenInheritable || !seenAmbient {
		return errors.New("launcher capability status is incomplete")
	}
	return nil
}

func configureSignerCommand(command *exec.Cmd, options runOptions) {
	attributes := &syscall.SysProcAttr{Pdeathsig: syscall.SIGKILL}
	if !options.allowLocalDev {
		attributes.Credential = &syscall.Credential{
			Uid:         options.signerUID,
			Gid:         options.signerGID,
			Groups:      []uint32{},
			NoSetGroups: false,
		}
	}
	command.SysProcAttr = attributes
}

func configureSupervisorCommand(command *exec.Cmd, options runOptions) {
	attributes := &syscall.SysProcAttr{
		Setpgid:   true,
		Pdeathsig: syscall.SIGKILL,
	}
	if !options.allowLocalDev {
		attributes.Credential = &syscall.Credential{
			Uid:         options.supervisorUID,
			Gid:         options.supervisorGID,
			Groups:      []uint32{},
			NoSetGroups: false,
		}
	}
	command.SysProcAttr = attributes
}

const (
	childExited  = 1
	childKilled  = 2
	childDumped  = 3
	childTrapped = 4
	childStopped = 5
)

func waitForSupervisorEvent(command *exec.Cmd) (supervisorEvent, error) {
	if command == nil || command.Process == nil {
		return supervisorEvent{}, errors.New("trusted signing supervisor was not started")
	}
	var processInfo unix.Siginfo
	for {
		err := unix.Waitid(
			unix.P_PID,
			command.Process.Pid,
			&processInfo,
			unix.WEXITED|unix.WSTOPPED|unix.WNOWAIT,
			nil,
		)
		if errors.Is(err, syscall.EINTR) {
			continue
		}
		if err != nil {
			return supervisorEvent{}, fmt.Errorf(
				"could not observe trusted signing supervisor: %w", err,
			)
		}
		break
	}
	event := supervisorEvent{code: processInfo.Code}
	switch processInfo.Code {
	case childExited, childKilled, childDumped:
		return event, nil
	case childTrapped, childStopped:
		event.stopped = true
		return event, nil
	default:
		return event, fmt.Errorf(
			"trusted signing supervisor produced unexpected waitid code %d",
			processInfo.Code,
		)
	}
}

func verifySupervisorProcessGroup(command *exec.Cmd) (bool, error) {
	if command == nil || command.Process == nil {
		return false, errors.New("trusted signing supervisor was not started")
	}
	pid := command.Process.Pid
	group, err := syscall.Getpgid(pid)
	if err != nil {
		return false, fmt.Errorf("could not verify trusted supervisor process group: %w", err)
	}
	if group != pid {
		return false, fmt.Errorf(
			"trusted supervisor process group is %d, expected %d", group, pid,
		)
	}
	return true, nil
}

func terminateSupervisorProcessGroup(command *exec.Cmd, groupVerified bool) error {
	if command == nil || command.Process == nil {
		return nil
	}
	var err error
	if groupVerified {
		err = syscall.Kill(-command.Process.Pid, syscall.SIGKILL)
	} else {
		err = command.Process.Kill()
	}
	if err != nil && !errors.Is(err, syscall.ESRCH) && !errors.Is(err, os.ErrProcessDone) {
		return fmt.Errorf("could not terminate trusted supervisor process group: %w", err)
	}
	return nil
}
