//go:build linux

package main

import (
	"errors"
	"fmt"
	"io"
	"os"
	"strconv"
	"strings"
	"syscall"
)

const (
	brokerPeerChallenge = 0xa5
	brokerPeerResponse  = 0x5a
)

const (
	prSetDumpable   = 4
	prSetNoNewPrivs = 38
)

func hardenProcess() error {
	if err := validatePlatformPrivilegeState(); err != nil {
		return err
	}
	if err := syscall.Setrlimit(syscall.RLIMIT_CORE, &syscall.Rlimit{Cur: 0, Max: 0}); err != nil {
		return err
	}
	if _, _, errno := syscall.Syscall6(
		syscall.SYS_PRCTL,
		uintptr(prSetDumpable),
		0,
		0,
		0,
		0,
		0,
	); errno != 0 {
		return fmt.Errorf("prctl(PR_SET_DUMPABLE): %w", errno)
	}
	if _, _, errno := syscall.Syscall6(
		syscall.SYS_PRCTL,
		uintptr(prSetNoNewPrivs),
		1,
		0,
		0,
		0,
		0,
	); errno != 0 {
		return fmt.Errorf("prctl(PR_SET_NO_NEW_PRIVS): %w", errno)
	}
	return nil
}

func validatePlatformPrivilegeState() error {
	raw, err := os.ReadFile("/proc/self/status")
	if err != nil {
		return fmt.Errorf("could not inspect Linux capability state: %w", err)
	}
	required := map[string]bool{"CapPrm": false, "CapEff": false, "CapAmb": false}
	for _, line := range strings.Split(string(raw), "\n") {
		name, value, found := strings.Cut(line, ":")
		if !found {
			continue
		}
		if _, wanted := required[name]; !wanted {
			continue
		}
		parsed, parseErr := strconv.ParseUint(strings.TrimSpace(value), 16, 64)
		if parseErr != nil {
			return fmt.Errorf("could not parse Linux %s capability state: %w", name, parseErr)
		}
		if parsed != 0 {
			return fmt.Errorf("signing supervisor must not carry Linux %s capabilities", name)
		}
		required[name] = true
	}
	for name, found := range required {
		if !found {
			return fmt.Errorf("Linux capability state is missing %s", name)
		}
	}
	return nil
}

func validatePlatformPathSecurity(path string) error {
	size, err := syscall.Getxattr(path, "security.capability", nil)
	if err == nil {
		if size > 0 {
			return fmt.Errorf("trusted path must not carry security.capability: %s", path)
		}
		return nil
	}
	if errors.Is(err, syscall.ENODATA) || errors.Is(err, syscall.ENOTSUP) {
		return nil
	}
	return fmt.Errorf("could not inspect security.capability on %s: %w", path, err)
}

func authenticateBrokerPeer(connection *os.File) error {
	challenge := []byte{0}
	if _, err := io.ReadFull(connection, challenge); err != nil {
		return err
	}
	if challenge[0] != brokerPeerChallenge {
		return errors.New("broker peer challenge is invalid")
	}
	_, err := connection.Write([]byte{brokerPeerResponse})
	return err
}
