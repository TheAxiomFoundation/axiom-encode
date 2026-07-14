//go:build darwin || linux

package main

import (
	"strings"
	"testing"
)

func envFrom(pairs map[string]string) func(string) string {
	return func(name string) string { return pairs[name] }
}

func ciBinding() contextBinding {
	return contextBinding{
		expectedRepository:  "TheAxiomFoundation/rulespec-uk",
		allowedWorkflowRefs: []string{"TheAxiomFoundation/rulespec-uk/.github/workflows/bulk-encode.yml@refs/heads/main"},
		allowedEventNames:   []string{"workflow_dispatch", "schedule"},
	}
}

func validCIEnvironment() map[string]string {
	return map[string]string{
		"GITHUB_ACTIONS":      "true",
		"GITHUB_REPOSITORY":   "TheAxiomFoundation/rulespec-uk",
		"GITHUB_WORKFLOW_REF": "TheAxiomFoundation/rulespec-uk/.github/workflows/bulk-encode.yml@refs/heads/main",
		"GITHUB_EVENT_NAME":   "workflow_dispatch",
		"GITHUB_SHA":          "abc123",
		"GITHUB_RUN_ID":       "42",
	}
}

func TestContextBindingAcceptsValidWorkflowDispatch(t *testing.T) {
	info, err := ciBinding().validate(envFrom(validCIEnvironment()))
	if err != nil {
		t.Fatalf("expected valid context, got %v", err)
	}
	if info.local || info.repository != "TheAxiomFoundation/rulespec-uk" || info.runID != "42" {
		t.Fatalf("unexpected context info: %+v", info)
	}
}

func TestContextBindingRefusals(t *testing.T) {
	cases := []struct {
		name     string
		mutate   func(map[string]string)
		binding  func(contextBinding) contextBinding
		contains string
	}{
		{
			name:     "not in actions",
			mutate:   func(env map[string]string) { env["GITHUB_ACTIONS"] = "false" },
			contains: "outside GitHub Actions",
		},
		{
			name:     "repository mismatch",
			mutate:   func(env map[string]string) { env["GITHUB_REPOSITORY"] = "attacker/fork" },
			contains: "does not match the expected repository",
		},
		{
			name: "workflow ref not allowlisted",
			mutate: func(env map[string]string) {
				env["GITHUB_WORKFLOW_REF"] = "TheAxiomFoundation/rulespec-uk/.github/workflows/bulk-encode.yml@refs/pull/9/merge"
			},
			contains: "not in the allowed workflow-ref list",
		},
		{
			name:     "event not allowlisted",
			mutate:   func(env map[string]string) { env["GITHUB_EVENT_NAME"] = "push" },
			contains: "not in the allowed event list",
		},
		{
			name:   "pull_request hard-refused even if allowlisted",
			mutate: func(env map[string]string) { env["GITHUB_EVENT_NAME"] = "pull_request" },
			binding: func(binding contextBinding) contextBinding {
				binding.allowedEventNames = append(binding.allowedEventNames, "pull_request")
				return binding
			},
			contains: "refuses to run on the \"pull_request\" event",
		},
		{
			name:   "pull_request_target hard-refused",
			mutate: func(env map[string]string) { env["GITHUB_EVENT_NAME"] = "pull_request_target" },
			binding: func(binding contextBinding) contextBinding {
				binding.allowedEventNames = append(binding.allowedEventNames, "pull_request_target")
				return binding
			},
			contains: "pull_request_target",
		},
	}
	for _, testCase := range cases {
		t.Run(testCase.name, func(t *testing.T) {
			environment := validCIEnvironment()
			testCase.mutate(environment)
			binding := ciBinding()
			if testCase.binding != nil {
				binding = testCase.binding(binding)
			}
			_, err := binding.validate(envFrom(environment))
			if err == nil || !strings.Contains(err.Error(), testCase.contains) {
				t.Fatalf("expected refusal containing %q, got %v", testCase.contains, err)
			}
		})
	}
}

func TestLocalDevRefusedInsideActions(t *testing.T) {
	binding := contextBinding{allowLocalDev: true}
	_, err := binding.validate(envFrom(map[string]string{"GITHUB_ACTIONS": "true"}))
	if err == nil || !strings.Contains(err.Error(), "refused inside GitHub Actions") {
		t.Fatalf("expected local-dev refusal inside Actions, got %v", err)
	}
}

func TestLocalDevAllowedOutsideActions(t *testing.T) {
	binding := contextBinding{allowLocalDev: true}
	info, err := binding.validate(envFrom(map[string]string{}))
	if err != nil || !info.local {
		t.Fatalf("expected local-dev context outside Actions, got info=%+v err=%v", info, err)
	}
}

func TestCIModeRequiresExplicitPolicyFlags(t *testing.T) {
	for _, missing := range []string{"repository", "refs", "events"} {
		binding := ciBinding()
		switch missing {
		case "repository":
			binding.expectedRepository = ""
		case "refs":
			binding.allowedWorkflowRefs = nil
		case "events":
			binding.allowedEventNames = nil
		}
		if _, err := binding.validate(envFrom(validCIEnvironment())); err == nil {
			t.Fatalf("expected CI mode to require the %q policy flag", missing)
		}
	}
}
