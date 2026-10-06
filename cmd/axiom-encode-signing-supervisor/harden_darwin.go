//go:build darwin

package main

import (
	"fmt"
	"os"
	"syscall"
)

const ptDenyAttach = 31

func hardenProcess() error {
	if err := syscall.Setrlimit(syscall.RLIMIT_CORE, &syscall.Rlimit{Cur: 0, Max: 0}); err != nil {
		return err
	}
	_, _, errno := syscall.Syscall6(
		syscall.SYS_PTRACE,
		uintptr(ptDenyAttach),
		0,
		0,
		0,
		0,
		0,
	)
	if errno != 0 {
		return fmt.Errorf("ptrace(PT_DENY_ATTACH): %w", errno)
	}
	return nil
}

func validatePlatformPrivilegeState() error {
	return nil
}

func validatePlatformPathSecurity(_ string) error {
	return nil
}

func authenticateBrokerPeer(_ *os.File) error {
	return nil
}
