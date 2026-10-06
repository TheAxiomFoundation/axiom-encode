//go:build linux

package main

import (
	"fmt"
	"syscall"
)

const (
	prSetDumpable   = 4
	prSetNoNewPrivs = 38
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
