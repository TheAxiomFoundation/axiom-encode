//go:build darwin || linux

package main

import (
	"errors"
	"fmt"
)

// contextBinding is the trusted-context policy passed as explicit flags by the
// workflow definition. The ambient GITHUB_* environment alone never defines what
// is acceptable: the launcher/signer refuses unless the ambient context matches
// the values the main-branch workflow text pinned.
type contextBinding struct {
	expectedRepository  string
	allowedWorkflowRefs []string
	allowedEventNames   []string
	allowLocalDev       bool
}

// contextInfo is the authenticated, non-secret context recorded for the audit
// log after a successful bind.
type contextInfo struct {
	local       bool
	repository  string
	workflowRef string
	eventName   string
	sha         string
	runID       string
}

// forbiddenEventNames can never sign, even if an operator mistakenly allowlists
// them. Pull-request events run with fork-controlled refs and are the primary
// exfiltration vector this boundary exists to close, so they are refused as a
// floor beneath the allowlist.
var forbiddenEventNames = map[string]struct{}{
	"pull_request":        {},
	"pull_request_target": {},
}

func containsString(values []string, target string) bool {
	for _, value := range values {
		if value == target {
			return true
		}
	}
	return false
}

// validate enforces the context policy against the ambient environment (read
// through getenv for testability) and returns the authenticated context.
func (binding contextBinding) validate(getenv func(string) string) (contextInfo, error) {
	inActions := getenv("GITHUB_ACTIONS") == "true"

	if binding.allowLocalDev {
		if inActions {
			return contextInfo{}, errors.New(
				"--allow-local-dev is refused inside GitHub Actions; the CI context " +
					"binding must not be bypassed where a real signing key is present",
			)
		}
		return contextInfo{local: true}, nil
	}

	if !inActions {
		return contextInfo{}, errors.New(
			"external signer refuses to run outside GitHub Actions without " +
				"--allow-local-dev (GITHUB_ACTIONS is not \"true\")",
		)
	}
	if binding.expectedRepository == "" {
		return contextInfo{}, errors.New("--expected-github-repository is required in CI mode")
	}
	if len(binding.allowedWorkflowRefs) == 0 {
		return contextInfo{}, errors.New("at least one --allowed-workflow-ref is required in CI mode")
	}
	if len(binding.allowedEventNames) == 0 {
		return contextInfo{}, errors.New("at least one --allowed-event-name is required in CI mode")
	}

	repository := getenv("GITHUB_REPOSITORY")
	workflowRef := getenv("GITHUB_WORKFLOW_REF")
	eventName := getenv("GITHUB_EVENT_NAME")

	if _, forbidden := forbiddenEventNames[eventName]; forbidden {
		return contextInfo{}, fmt.Errorf(
			"external signer refuses to run on the %q event; the signing leg must "+
				"run only on main-branch workflow_dispatch or schedule", eventName,
		)
	}
	if repository != binding.expectedRepository {
		return contextInfo{}, fmt.Errorf(
			"GITHUB_REPOSITORY %q does not match the expected repository %q",
			repository, binding.expectedRepository,
		)
	}
	if !containsString(binding.allowedWorkflowRefs, workflowRef) {
		return contextInfo{}, fmt.Errorf(
			"GITHUB_WORKFLOW_REF %q is not in the allowed workflow-ref list", workflowRef,
		)
	}
	if !containsString(binding.allowedEventNames, eventName) {
		return contextInfo{}, fmt.Errorf(
			"GITHUB_EVENT_NAME %q is not in the allowed event list", eventName,
		)
	}

	return contextInfo{
		repository:  repository,
		workflowRef: workflowRef,
		eventName:   eventName,
		sha:         getenv("GITHUB_SHA"),
		runID:       getenv("GITHUB_RUN_ID"),
	}, nil
}
