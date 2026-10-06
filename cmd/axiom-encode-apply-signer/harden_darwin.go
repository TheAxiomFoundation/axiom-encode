//go:build darwin

package main

import (
	"fmt"
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
