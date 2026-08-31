//go:build darwin

package main

import (
	"os/exec"
	"syscall"
)

func prepareDescendantContainment() error {
	return nil
}

func configureContainedCommand(command *exec.Cmd) {
	if command.SysProcAttr == nil {
		command.SysProcAttr = &syscall.SysProcAttr{}
	}
	command.SysProcAttr.Setpgid = true
}

func waitContainedCommand(command *exec.Cmd, _ int) (error, error) {
	// macOS has no Linux-style child-subreaper facility. Keeping the supervisor
	// resident still preserves child status and the broker lifecycle; Linux adds
	// the stronger process-tree drain used by the production runner.
	return command.Wait(), nil
}

func drainContainedDescendants(_ int) error {
	return nil
}
