//go:build darwin

package main

import (
	"errors"
	"fmt"
	"os"
	"os/exec"
	"syscall"
)

const ptDenyAttach = 31

// hardenProcess denies core dumps and same-user debugger attachment for the
// process holding (or briefly handling) private key material.
func hardenProcess() error {
	if err := syscall.Setrlimit(syscall.RLIMIT_CORE, &syscall.Rlimit{Cur: 0, Max: 0}); err != nil {
		return err
	}
	if _, _, errno := syscall.Syscall6(
		syscall.SYS_PTRACE, uintptr(ptDenyAttach), 0, 0, 0, 0, 0,
	); errno != 0 {
		return fmt.Errorf("ptrace(PT_DENY_ATTACH): %w", errno)
	}
	return nil
}

func validateLauncherSecurity(options runOptions) error {
	if !options.allowLocalDev {
		return errors.New("production run requires the Linux privilege-separated launcher")
	}
	if os.Getuid() == 0 || os.Geteuid() == 0 || os.Getgid() == 0 || os.Getegid() == 0 {
		return errors.New("--allow-local-dev is refused for a root launcher")
	}
	return nil
}

func validateSupervisorExecutable(options runOptions) (string, error) {
	if !options.allowLocalDev {
		return "", errors.New("production supervisor validation requires Linux")
	}
	return options.supervisor, nil
}

func configureSignerCommand(command *exec.Cmd, _ runOptions) {
	command.SysProcAttr = &syscall.SysProcAttr{}
}

func configureSupervisorCommand(command *exec.Cmd, _ runOptions) {
	command.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
}

// Production is Linux-only. The Darwin launcher path exists solely for local
// protocol tests, so ordinary Cmd.Wait is sufficient and cannot weaken CI.
func waitForSupervisorEvent(command *exec.Cmd) (supervisorEvent, error) {
	if command == nil || command.Process == nil {
		return supervisorEvent{}, errors.New("trusted signing supervisor was not started")
	}
	waitErr := command.Wait()
	return supervisorEvent{reaped: true, waitErr: waitErr}, nil
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
