//go:build darwin || linux

package main

import (
	"bytes"
	"crypto/ed25519"
	"crypto/rand"
	"encoding/base64"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

func subscriptionLauncherArguments(supervisor string, subscription []string) []string {
	arguments := []string{
		"run",
		"--scope", "apply_ed25519",
		"--key-env", "APPLY_SIGNER_SUBSCRIPTION_TEST_KEY",
		"--supervisor", supervisor,
		"--trusted-signing-roots", "/protected roots/trust.json",
		"--trusted-python-runtime-root", "/protected python/runtime",
		"--trusted-python-import-root", "/protected python/import one",
		"--trusted-python-import-root", "/protected python/import two",
		"--trusted-python-package-root", "/protected python/axiom_encode",
		"--expected-github-repository", "TheAxiomFoundation/rulespec-uk",
		"--allowed-workflow-ref", "TheAxiomFoundation/rulespec-uk/.github/workflows/bulk-encode.yml@refs/heads/main",
		"--allowed-event-name", "workflow_dispatch",
	}
	arguments = append(arguments, subscription...)
	return append(arguments,
		"--", "/protected encoder/axiom-encode", "encode", "us/statute/toy",
		"--backend", "codex", "--apply",
	)
}

func subscriptionFlagSubset(mask int, paths []string) []string {
	flags := []string{"--trusted-codex-cli-config", "--codex-subscription-auth", "--codex-auth-outbox"}
	var arguments []string
	for index, flag := range flags {
		if mask&(1<<index) != 0 {
			arguments = append(arguments, flag, paths[index])
		}
	}
	return arguments
}

func TestParseRunSubscriptionOptions(t *testing.T) {
	paths := []string{"/protected config/codex.json", "/operator auth/auth.json", "/operator outbox/auth.json"}
	for mask := 0; mask < 8; mask++ {
		t.Run(fmt.Sprintf("subset_%03b", mask), func(t *testing.T) {
			arguments := subscriptionLauncherArguments("/protected supervisor", subscriptionFlagSubset(mask, paths))
			_, err := parseRunOptions(arguments[1:])
			if mask == 0 || mask == 7 {
				if err != nil {
					t.Fatalf("complete or omitted subscription options must parse: %v", err)
				}
				return
			}
			want := "Codex outbox/config requires --codex-subscription-auth"
			if mask&2 != 0 {
				want = "Codex subscription auth requires --codex-auth-outbox and --trusted-codex-cli-config"
			}
			if err == nil || err.Error() != want {
				t.Fatalf("partial subscription options must fail with %q, got %v", want, err)
			}
		})
	}
}

func TestLauncherSubscriptionForwardsExactArguments(t *testing.T) {
	_, private, err := ed25519.GenerateKey(rand.Reader)
	if err != nil {
		t.Fatal(err)
	}
	seedB64 := base64.StdEncoding.EncodeToString(private.Seed())
	fixtureRoot := t.TempDir()
	paths := []string{
		filepath.Join(fixtureRoot, "protected config ' $ literal.json"),
		filepath.Join(fixtureRoot, "operator auth ü.json"),
		filepath.Join(fixtureRoot, "refreshed auth ; literal.json"),
	}
	// Invented data only: the launcher must pass paths, never open these files.
	const authSecret = "invented-subscription-auth-only"
	const configMarker = "invented-codex-config-only"
	authContents := []byte(`{"access_token":"` + authSecret + `"}`)
	configContents := []byte(`{"executable":"` + configMarker + `"}`)
	if err := os.WriteFile(paths[0], configContents, 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(paths[1], authContents, 0o600); err != nil {
		t.Fatal(err)
	}
	for _, enabled := range []bool{false, true} {
		t.Run(fmt.Sprintf("subscription_%t", enabled), func(t *testing.T) {
			dump := filepath.Join(t.TempDir(), "supervisor-argv.bin")
			environmentDump := filepath.Join(t.TempDir(), "supervisor-environment.bin")
			supervisor := filepath.Join(t.TempDir(), "supervisor with spaces")
			script := "#!/bin/sh\n" +
				"printf '%s\\0' \"$@\" > " + shellQuote(dump) + "\n" +
				"env > " + shellQuote(environmentDump) + "\n"
			if err := os.WriteFile(supervisor, []byte(script), 0o755); err != nil {
				t.Fatal(err)
			}
			var subscription []string
			if enabled {
				subscription = subscriptionFlagSubset(7, paths)
			}
			command := exec.Command(applySignerBinary, subscriptionLauncherArguments(supervisor, subscription)...)
			command.Env = append(ciEnvironmentPairs(), "APPLY_SIGNER_SUBSCRIPTION_TEST_KEY="+seedB64)
			output, launchErr := command.CombinedOutput()
			if bytes.Contains(output, []byte(seedB64)) || bytes.Contains(output, []byte(authSecret)) || bytes.Contains(output, []byte(configMarker)) {
				t.Fatal("launcher output leaked test key or file contents")
			}
			if launchErr != nil {
				t.Fatalf("launcher failed: %v\n%s", launchErr, output)
			}
			raw, err := os.ReadFile(dump)
			if err != nil {
				t.Fatalf("supervisor stub did not run: %v", err)
			}
			environment, err := os.ReadFile(environmentDump)
			if err != nil {
				t.Fatal(err)
			}
			for _, record := range [][]byte{raw, environment} {
				for _, forbidden := range [][]byte{[]byte(seedB64), []byte("APPLY_SIGNER_SUBSCRIPTION_TEST_KEY"), []byte(authSecret), []byte(configMarker)} {
					if bytes.Contains(record, forbidden) {
						t.Fatal("supervisor received test key or file contents in argv/environment")
					}
				}
			}
			want := []string{
				"--apply-signer-fd", "3",
				"--trusted-signing-roots", "/protected roots/trust.json",
				"--trusted-python-runtime-root", "/protected python/runtime",
				"--trusted-python-import-root", "/protected python/import one",
				"--trusted-python-import-root", "/protected python/import two",
				"--trusted-python-package-root", "/protected python/axiom_encode",
			}
			want = append(want, subscription...)
			want = append(want, "--", "/protected encoder/axiom-encode", "encode", "us/statute/toy", "--backend", "codex", "--apply")
			if !bytes.Equal(raw, []byte(strings.Join(want, "\x00")+"\x00")) {
				t.Fatalf("supervisor argv mismatch:\n got: %q\nwant: %q", raw, want)
			}
			for index, contents := range [][]byte{configContents, authContents} {
				got, err := os.ReadFile(paths[index])
				if err != nil || !bytes.Equal(got, contents) {
					t.Fatal("launcher modified subscription input file")
				}
			}
			if _, err := os.Stat(paths[2]); !os.IsNotExist(err) {
				t.Fatalf("launcher unexpectedly created auth outbox: %v", err)
			}
		})
	}
}

func TestLauncherSubscriptionPreservesCIRestriction(t *testing.T) {
	paths := []string{"/protected config/codex.json", "/operator auth/auth.json", "/operator outbox/auth.json"}
	command := exec.Command(applySignerBinary, subscriptionLauncherArguments("/bin/true", subscriptionFlagSubset(7, paths))...)
	const testKey = "invented-key-must-not-be-consumed"
	command.Env = []string{"PATH=" + os.Getenv("PATH"), "APPLY_SIGNER_SUBSCRIPTION_TEST_KEY=" + testKey}
	output, err := command.CombinedOutput()
	if bytes.Contains(output, []byte(testKey)) {
		t.Fatal("launcher refusal leaked test key")
	}
	if err == nil || !strings.Contains(string(output), "outside GitHub Actions") {
		t.Fatalf("subscription options must preserve CI restriction: error=%v output=%s", err, output)
	}
}

func TestRunLauncherSubscriptionRejectsBeforeConsumingKey(t *testing.T) {
	for mask := 1; mask < 7; mask++ {
		t.Run(fmt.Sprintf("subset_%03b", mask), func(t *testing.T) {
			const keyName = "APPLY_SIGNER_SUBSCRIPTION_TEST_KEY"
			const testKey = "invented-key-must-not-be-consumed"
			t.Setenv(keyName, testKey)
			options := runOptions{scope: scopeApply, keyEnv: keyName, binding: ciBinding()}
			if mask&1 != 0 {
				options.codexCLIConfigPath = "/protected config/codex.json"
			}
			if mask&2 != 0 {
				options.codexAuthPath = "/operator auth/auth.json"
			}
			if mask&4 != 0 {
				options.codexAuthOutbox = "/operator outbox/auth.json"
			}
			_, err := runLauncher(options, envFrom(validCIEnvironment()))
			want := "Codex outbox/config requires --codex-subscription-auth"
			if mask&2 != 0 {
				want = "Codex subscription auth requires --codex-auth-outbox and --trusted-codex-cli-config"
			}
			if err == nil || err.Error() != want {
				t.Fatalf("partial subscription options must fail before key handling: %v", err)
			}
			if os.Getenv(keyName) != testKey {
				t.Fatal("partial subscription options consumed the test key")
			}
		})
	}
}

func TestSubscriptionCredentialsExcludedFromSignerEnvironment(t *testing.T) {
	environment := validCIEnvironment()
	environment["CODEX_HOME"] = "/invented operator/home"
	environment["CODEX_AUTH_SOURCE"] = "/invented operator/auth.json"
	environment["CODEX_AUTH_OUTBOX"] = "/invented operator/outbox.json"
	environment["APPLY_SIGNER_SUBSCRIPTION_TEST_KEY"] = "invented-private-key"
	environment["OPENAI_API_KEY"] = "invented-api-key"
	got := minimalSignerEnvironment(envFrom(environment))
	want := []string{
		"GITHUB_ACTIONS=true",
		"GITHUB_REPOSITORY=TheAxiomFoundation/rulespec-uk",
		"GITHUB_WORKFLOW_REF=TheAxiomFoundation/rulespec-uk/.github/workflows/bulk-encode.yml@refs/heads/main",
		"GITHUB_EVENT_NAME=workflow_dispatch",
		"GITHUB_SHA=abc123",
		"GITHUB_RUN_ID=42",
	}
	if strings.Join(got, "\x00") != strings.Join(want, "\x00") {
		t.Fatal("signer environment must remain the six CI context fields only")
	}
}
