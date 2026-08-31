//go:build linux

package main

import (
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"syscall"

	"golang.org/x/sys/unix"
)

const prSetChildSubreaper = 36

func prepareDescendantContainment() error {
	if _, _, errno := syscall.Syscall6(
		syscall.SYS_PRCTL,
		uintptr(prSetChildSubreaper),
		1,
		0,
		0,
		0,
		0,
	); errno != 0 {
		return fmt.Errorf("prctl(PR_SET_CHILD_SUBREAPER): %w", errno)
	}
	return nil
}

func configureContainedCommand(command *exec.Cmd) {
	if command.SysProcAttr == nil {
		command.SysProcAttr = &syscall.SysProcAttr{}
	}
	command.SysProcAttr.Setpgid = true
}

func waitContainedCommand(command *exec.Cmd, brokerPID int) (error, error) {
	if command.Process == nil {
		return nil, errors.New("trusted command was not started")
	}
	leaderPID := command.Process.Pid
	var cleanupErrors []error
	groupVerified := false
	if processGroup, err := syscall.Getpgid(leaderPID); err != nil {
		cleanupErrors = append(
			cleanupErrors,
			fmt.Errorf("could not verify trusted command process group: %w", err),
		)
	} else if processGroup != leaderPID {
		cleanupErrors = append(
			cleanupErrors,
			fmt.Errorf(
				"trusted command process group is %d, expected %d",
				processGroup,
				leaderPID,
			),
		)
	} else {
		groupVerified = true
	}

	leaderObserved := false
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
		if err != nil {
			cleanupErrors = append(
				cleanupErrors,
				fmt.Errorf("could not observe trusted command exit without reaping: %w", err),
			)
		} else {
			leaderObserved = true
		}
		break
	}

	// Keep the exited leader as a zombie until after the negative-PID kill. Its
	// unreaped PID pins the process-group identity, so a recycled PID can never
	// redirect the signal to an unrelated process group.
	if groupVerified && leaderObserved {
		if err := syscall.Kill(-leaderPID, syscall.SIGKILL); err != nil &&
			!errors.Is(err, syscall.ESRCH) {
			cleanupErrors = append(
				cleanupErrors,
				fmt.Errorf("could not terminate trusted command process group: %w", err),
			)
		}
	}

	waitErr := command.Wait()
	var commandErr error
	if waitErr != nil {
		var exitError *exec.ExitError
		if errors.As(waitErr, &exitError) {
			commandErr = waitErr
		} else {
			cleanupErrors = append(
				cleanupErrors,
				fmt.Errorf("could not reap trusted command leader: %w", waitErr),
			)
		}
	}
	if err := drainContainedDescendants(brokerPID); err != nil {
		cleanupErrors = append(cleanupErrors, err)
	}
	return commandErr, errors.Join(cleanupErrors...)
}

func drainContainedDescendants(excludedPID int) error {
	// Each pass kills and reaps every direct child other than the known broker.
	// Reaping exposes the next layer of a double-fork tree to this subreaper, so
	// repeated passes also catch descendants that escaped the original group with
	// setsid(2).
	const maximumDrainPasses = 4096
	for pass := 0; pass < maximumDrainPasses; pass++ {
		children, err := directContainedChildren(excludedPID)
		if err != nil {
			return err
		}
		if len(children) == 0 {
			return nil
		}
		for _, childPID := range children {
			if err := terminateAndReapContainedChild(childPID); err != nil {
				return err
			}
		}
	}
	return errors.New("descendant drain exceeded its process-tree pass limit")
}

func directContainedChildren(excludedPID int) ([]int, error) {
	processes, err := os.ReadDir("/proc")
	if err != nil {
		return nil, fmt.Errorf("could not enumerate Linux processes: %w", err)
	}
	supervisorPID := os.Getpid()
	children := make(map[int]struct{})
	for _, process := range processes {
		pid, parseErr := strconv.Atoi(process.Name())
		if parseErr != nil || pid <= 0 {
			continue
		}
		raw, readErr := os.ReadFile(filepath.Join("/proc", process.Name(), "status"))
		if errors.Is(readErr, os.ErrNotExist) {
			// The process exited between ReadDir and ReadFile.
			continue
		}
		if errors.Is(readErr, os.ErrPermission) {
			// hidepid can conceal unrelated users' status files. A descendant of
			// this no-capability supervisor cannot change to an inaccessible UID.
			continue
		}
		if readErr != nil {
			return nil, fmt.Errorf("could not inspect Linux process %d: %w", pid, readErr)
		}
		parentPID := -1
		for _, line := range strings.Split(string(raw), "\n") {
			fields := strings.Fields(line)
			if len(fields) == 2 && fields[0] == "PPid:" {
				parentPID, parseErr = strconv.Atoi(fields[1])
				if parseErr != nil || parentPID < 0 {
					return nil, fmt.Errorf(
						"kernel returned invalid parent PID %q for process %d",
						fields[1],
						pid,
					)
				}
				break
			}
		}
		if parentPID < 0 {
			return nil, fmt.Errorf("Linux process %d status is missing PPid", pid)
		}
		if parentPID == supervisorPID && pid != excludedPID {
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

func terminateAndReapContainedChild(pid int) error {
	var status syscall.WaitStatus
	for {
		waitedPID, err := syscall.Wait4(
			pid,
			&status,
			syscall.WNOHANG|syscall.WALL,
			nil,
		)
		if errors.Is(err, syscall.EINTR) {
			continue
		}
		if errors.Is(err, syscall.ECHILD) {
			// The task-level children file can become stale while a thread exits.
			// Never signal a PID once the kernel says it is not our child.
			return nil
		}
		if err != nil {
			return fmt.Errorf("could not inspect adopted child %d: %w", pid, err)
		}
		if waitedPID == pid {
			return nil
		}
		if waitedPID != 0 {
			return fmt.Errorf("wait4 returned unexpected child PID %d for %d", waitedPID, pid)
		}
		break
	}

	// A running direct child retains its PID until this process reaps it. Even if
	// it exits between WNOHANG and kill, it becomes a zombie and the numeric PID
	// cannot be recycled to an unrelated process.
	if err := syscall.Kill(pid, syscall.SIGKILL); err != nil &&
		!errors.Is(err, syscall.ESRCH) {
		return fmt.Errorf("could not terminate adopted child %d: %w", pid, err)
	}
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
