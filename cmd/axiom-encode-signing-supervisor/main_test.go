//go:build darwin || linux

package main

import (
	"bytes"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"reflect"
	"strconv"
	"strings"
	"testing"

	"golang.org/x/sys/unix"
)

func encodedTestPublicKey(value byte) string {
	return base64.StdEncoding.EncodeToString(bytes.Repeat([]byte{value}, 32))
}

func parseTestTrustRoots(t *testing.T, payload map[string]any) ([]byte, []byte, [][]byte, error) {
	t.Helper()
	raw, err := json.Marshal(payload)
	if err != nil {
		t.Fatal(err)
	}
	apply, eval, corpus, _, _, err := parseProtectedTrustRoots(raw, "/protected/signing-trust-roots.json")
	return apply, eval, corpus, err
}

func baseTrustRoots() map[string]any {
	return map[string]any{
		"schema":                            "axiom-encode/signing-trust-roots/v2",
		"apply_ed25519_public_key":          encodedTestPublicKey(0xab),
		"eval_ed25519_public_key":           encodedTestPublicKey(0xcd),
		"corpus_release_ed25519_public_key": encodedTestPublicKey(0x17),
	}
}

func TestLoadProtectedTrustRootsAcceptsV2AndV3Keyring(t *testing.T) {
	v2 := baseTrustRoots()
	apply, eval, corpusKeys, err := parseTestTrustRoots(t, v2)
	if err != nil {
		t.Fatalf("v2 trust roots were rejected: %v", err)
	}
	if len(apply) != 32 || len(eval) != 32 || len(corpusKeys) != 1 || len(corpusKeys[0]) != 32 {
		t.Fatalf("v2 trust roots returned an invalid keyring: %#v", corpusKeys)
	}
	v2CorpusKey := append([]byte(nil), corpusKeys[0]...)

	v3Single := baseTrustRoots()
	v3Single["schema"] = "axiom-encode/signing-trust-roots/v3"
	delete(v3Single, "corpus_release_ed25519_public_key")
	v3Single["corpus_release_ed25519_public_keys"] = []string{
		encodedTestPublicKey(0x17),
	}
	_, _, corpusKeys, err = parseTestTrustRoots(t, v3Single)
	if err != nil {
		t.Fatalf("single-key v3 trust roots were rejected: %v", err)
	}
	if len(corpusKeys) != 1 || !bytes.Equal(corpusKeys[0], v2CorpusKey) {
		t.Fatalf("single-key v3 was not equivalent to v2: %#v", corpusKeys)
	}

	v3Consistent := baseTrustRoots()
	v3Consistent["schema"] = "axiom-encode/signing-trust-roots/v3"
	v3Consistent["corpus_release_ed25519_public_keys"] = []string{
		encodedTestPublicKey(0x17),
	}
	if _, _, _, err = parseTestTrustRoots(t, v3Consistent); err != nil {
		t.Fatalf("consistent singular/plural v3 trust roots were rejected: %v", err)
	}

	v3 := baseTrustRoots()
	v3["schema"] = "axiom-encode/signing-trust-roots/v3"
	delete(v3, "corpus_release_ed25519_public_key")
	v3["corpus_release_ed25519_public_keys"] = []string{
		encodedTestPublicKey(0x18),
		encodedTestPublicKey(0x17),
	}
	_, _, corpusKeys, err = parseTestTrustRoots(t, v3)
	if err != nil {
		t.Fatalf("v3 trust roots were rejected: %v", err)
	}
	if len(corpusKeys) != 2 || corpusKeys[0][0] != 0x18 || corpusKeys[1][0] != 0x17 {
		t.Fatalf("v3 trust roots lost keyring order: %#v", corpusKeys)
	}
}

func TestLoadProtectedTrustRootsRejectsInvalidV3Keyrings(t *testing.T) {
	for _, testCase := range []struct {
		name   string
		mutate func(map[string]any)
		error  string
	}{
		{
			name: "empty",
			mutate: func(payload map[string]any) {
				payload["corpus_release_ed25519_public_keys"] = []string{}
			},
			error: "must not be empty",
		},
		{
			name: "malformed base64",
			mutate: func(payload map[string]any) {
				payload["corpus_release_ed25519_public_keys"] = []string{"not-base64!!"}
			},
			error: "base64-encoded raw bytes",
		},
		{
			name: "wrong length",
			mutate: func(payload map[string]any) {
				payload["corpus_release_ed25519_public_keys"] = []string{base64.StdEncoding.EncodeToString([]byte("short"))}
			},
			error: "must contain 32 raw bytes",
		},
		{
			name: "conflicting singular and plural",
			mutate: func(payload map[string]any) {
				payload["corpus_release_ed25519_public_key"] = encodedTestPublicKey(0x19)
			},
			error: "singular and plural corpus release public keys conflict",
		},
		{
			name: "unknown schema",
			mutate: func(payload map[string]any) {
				payload["schema"] = "axiom-encode/signing-trust-roots/v999"
			},
			error: "schema is unsupported",
		},
	} {
		t.Run(testCase.name, func(t *testing.T) {
			payload := baseTrustRoots()
			payload["schema"] = "axiom-encode/signing-trust-roots/v3"
			payload["corpus_release_ed25519_public_keys"] = []string{encodedTestPublicKey(0x18)}
			delete(payload, "corpus_release_ed25519_public_key")
			testCase.mutate(payload)
			_, _, _, err := parseTestTrustRoots(t, payload)
			if err == nil || !strings.Contains(err.Error(), testCase.error) {
				t.Fatalf("expected %q rejection, got %v", testCase.error, err)
			}
		})
	}
}

func TestCredentialOutboxDestinationRejectsDevice(t *testing.T) {
	fd, err := unix.Open("/dev", unix.O_RDONLY|unix.O_DIRECTORY|unix.O_CLOEXEC, 0)
	if err != nil {
		t.Fatal(err)
	}
	defer unix.Close(fd)
	if err := validateCredentialOutboxDestination(fd, "null"); err == nil || !strings.Contains(err.Error(), "special file") {
		t.Fatalf("device destination was not rejected: %v", err)
	}
}

func TestCleanChildEnvironmentForwardsApplyCheckoutIdentity(t *testing.T) {
	values := map[string]string{
		"AXIOM_ENCODE_APPLY_CHECKOUT": "/home/runner/work/repo/repo/axiom-encode",
		"GITHUB_ACTIONS":              "true",
		"GITHUB_SHA":                  strings.Repeat("a", 40),
		"GITHUB_WORKSPACE":            "/home/runner/work/repo/repo",
		"UNTRUSTED_AMBIENT_VALUE":     "must-not-cross",
	}
	for name, value := range values {
		t.Setenv(name, value)
	}
	environment := cleanChildEnvironment(10, 11, "/trusted/bin", "/trusted", "")
	joined := "\n" + strings.Join(environment, "\n") + "\n"
	for name, value := range values {
		entry := "\n" + name + "=" + value + "\n"
		if name == "UNTRUSTED_AMBIENT_VALUE" {
			if strings.Contains(joined, entry) {
				t.Fatalf("untrusted ambient value crossed supervisor boundary")
			}
			continue
		}
		if !strings.Contains(joined, entry) {
			t.Fatalf("missing apply identity environment %s", name)
		}
	}
	if _, present := os.LookupEnv("AXIOM_ENCODE_APPLY_SIGNING_KEY"); present {
		t.Fatal("test environment unexpectedly contains an apply signing key")
	}
}

func framed(raw string) []byte {
	header := make([]byte, 4)
	binary.BigEndian.PutUint32(header, uint32(len(raw)))
	return append(header, []byte(raw)...)
}

func TestReceiveFrameRejectsTrailingJSONValue(t *testing.T) {
	var request brokerRequest
	err := receiveFrame(
		bytes.NewReader(framed(`{"version":4,"id":1,"op":"status"} {}`)),
		&request,
	)
	if err == nil || !strings.Contains(err.Error(), "trailing JSON") {
		t.Fatalf("expected trailing JSON rejection, got %v", err)
	}
}

func TestReceiveFrameRejectsUnknownField(t *testing.T) {
	var request brokerRequest
	err := receiveFrame(
		bytes.NewReader(framed(`{"version":4,"id":1,"op":"status","legacy_key":"no"}`)),
		&request,
	)
	if err == nil || !strings.Contains(err.Error(), "unknown field") {
		t.Fatalf("expected unknown field rejection, got %v", err)
	}
}

func TestReceiveFrameRejectsDuplicateKeysAtEveryDepth(t *testing.T) {
	for _, raw := range []string{
		`{"version":4,"id":1,"id":2,"op":"status"}`,
		`{"version":4,"\u0069d":1,"id":2,"op":"status"}`,
		`{"version":4,"id":1,"op":"status","payload":{"x":1,"x":2}}`,
	} {
		var request brokerRequest
		err := receiveFrame(bytes.NewReader(framed(raw)), &request)
		if err == nil || !strings.Contains(err.Error(), "duplicate object key") {
			t.Fatalf("expected duplicate-key rejection for %s, got %v", raw, err)
		}
	}
}

func TestReceiveFrameRejectsOversizedObjectKey(t *testing.T) {
	raw := `{"` + strings.Repeat("x", maxJSONKeyBytes+1) + `":1}`
	var request brokerRequest
	err := receiveFrame(bytes.NewReader(framed(raw)), &request)
	if err == nil || !strings.Contains(err.Error(), "object key exceeds") {
		t.Fatalf("expected oversized-key rejection, got %v", err)
	}
}

func TestReceiveFrameRejectsExcessiveJSONNesting(t *testing.T) {
	raw := `{"version":4,"id":1,"op":"status","payload":` +
		strings.Repeat("[", maxJSONDepth+2) + `null` +
		strings.Repeat("]", maxJSONDepth+2) + `}`
	var request brokerRequest
	err := receiveFrame(bytes.NewReader(framed(raw)), &request)
	if err == nil || !strings.Contains(err.Error(), "nesting exceeds") {
		t.Fatalf("expected excessive-nesting rejection, got %v", err)
	}
}

func decodedRequest(t *testing.T, raw string) brokerRequest {
	t.Helper()
	var request brokerRequest
	if err := receiveFrame(bytes.NewReader(framed(raw)), &request); err != nil {
		t.Fatalf("could not decode request: %v", err)
	}
	return request
}

func TestBrokerRequestRequiresExactOperationFields(t *testing.T) {
	testCases := []struct {
		name  string
		raw   string
		error string
	}{
		{
			name:  "empty payload on status",
			raw:   `{"version":4,"id":1,"op":"status","payload":""}`,
			error: "status request is malformed",
		},
		{
			name:  "null payload on sign",
			raw:   `{"version":4,"id":1,"op":"apply_ed25519_sign","payload":null}`,
			error: "signing request is malformed",
		},
		{
			name:  "missing payload on sign",
			raw:   `{"version":4,"id":1,"op":"apply_ed25519_sign"}`,
			error: "signing request is malformed",
		},
		{
			name:  "empty forbidden initialization field",
			raw:   `{"version":4,"id":1,"op":"status","apply_public_key":""}`,
			error: "forbidden initialization fields",
		},
	}
	for _, testCase := range testCases {
		t.Run(testCase.name, func(t *testing.T) {
			request := decodedRequest(t, testCase.raw)
			err := validateBrokerRequestOperation(&request)
			if err == nil || !strings.Contains(err.Error(), testCase.error) {
				t.Fatalf("expected %q rejection, got %v", testCase.error, err)
			}
		})
	}

	request := decodedRequest(
		t,
		`{"version":4,"id":1,"op":"apply_ed25519_sign","payload":""}`,
	)
	if err := validateBrokerRequestOperation(&request); err != nil {
		t.Fatalf("explicit empty signing payload should be valid: %v", err)
	}
}

func TestBrokerRequestIDsArePositiveAndStrictlyIncreasing(t *testing.T) {
	lastRequestID := int64(0)
	first := decodedRequest(t, `{"version":4,"id":1,"op":"status"}`)
	if err := validateBrokerRequestEnvelope(&first, &lastRequestID); err != nil {
		t.Fatalf("first request should be valid: %v", err)
	}
	for _, raw := range []string{
		`{"version":4,"id":1,"op":"status"}`,
		`{"version":4,"id":0,"op":"status"}`,
		`{"version":4,"id":-1,"op":"status"}`,
	} {
		request := decodedRequest(t, raw)
		err := validateBrokerRequestEnvelope(&request, &lastRequestID)
		if err == nil || !strings.Contains(err.Error(), "strictly increasing") {
			t.Fatalf("expected request ID rejection for %s, got %v", raw, err)
		}
	}
	third := decodedRequest(t, `{"version":4,"id":3,"op":"status"}`)
	if err := validateBrokerRequestEnvelope(&third, &lastRequestID); err != nil {
		t.Fatalf("later increasing request should be valid: %v", err)
	}
}

func TestBrokerRejectsRemovedV3Protocol(t *testing.T) {
	request := decodedRequest(t, `{"version":3,"id":1,"op":"status"}`)
	lastRequestID := int64(0)
	err := validateBrokerRequestEnvelope(&request, &lastRequestID)
	if err == nil || !strings.Contains(err.Error(), "malformed") {
		t.Fatalf("expected removed protocol rejection, got %v", err)
	}
}

func TestBrokerInitializationRejectsEmptyOrExtraFields(t *testing.T) {
	for _, raw := range []string{
		`{"version":4,"id":0,"op":"initialize","apply_public_key":""}`,
		`{"version":4,"id":0,"op":"initialize","apply_public_key":"q6urq6urq6urq6urq6urq6urq6urq6urq6urq6urq6s=","payload":""}`,
		`{"version":4,"id":0,"op":"initialize"}`,
	} {
		request := decodedRequest(t, raw)
		if request.hasValidInitializationShape() {
			t.Fatalf("expected initialization shape rejection for %s", raw)
		}
	}
}

func TestBrokerInitializationRequiresThreeDistinctRoots(t *testing.T) {
	valid := decodedRequest(
		t,
		`{"version":4,"id":0,"op":"initialize","apply_public_key":"q6urq6urq6urq6urq6urq6urq6urq6urq6urq6urq6s=","eval_public_key":"zc3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc0=","corpus_release_public_key":"FxcXFxcXFxcXFxcXFxcXFxcXFxcXFxcXFxcXFxcXFxc=","corpus_release_public_keys":["FxcXFxcXFxcXFxcXFxcXFxcXFxcXFxcXFxcXFxcXFxc="]}`,
	)
	if !valid.hasValidInitializationShape() {
		t.Fatal("expected three distinct initialization roots to be valid")
	}

	aliased := valid
	aliased.CorpusReleasePublicKey = append([]byte(nil), valid.ApplyPublicKey...)
	if aliased.hasValidInitializationShape() {
		t.Fatal("expected aliased corpus release root to be rejected")
	}

	conflicting := valid
	conflicting.CorpusReleasePublicKeys = [][]byte{bytes.Repeat([]byte{0x18}, 32)}
	if conflicting.hasValidInitializationShape() {
		t.Fatal("expected conflicting corpus release keyring to be rejected")
	}
}

func TestParseRawPublicKey(t *testing.T) {
	publicKey := bytes.Repeat([]byte{0xab}, 32)
	encoded := []byte("q6urq6urq6urq6urq6urq6urq6urq6urq6urq6urq6s=")
	parsed, err := parsePublicKey(encoded)
	if err != nil {
		t.Fatal(err)
	}
	defer zero(parsed)
	if !bytes.Equal(parsed, publicKey) {
		t.Fatal("parsed public key did not match")
	}
}

func TestBrokerAllowsVerificationOnlyButRejectsAliasedSignerDescriptors(t *testing.T) {
	parsed, err := parseBrokerOptions(nil)
	if err != nil || parsed.applySignerFD != -1 || parsed.evalSignerFD != -1 {
		t.Fatalf("expected verification-only broker options, got %#v err=%v", parsed, err)
	}
	_, err = parseBrokerOptions([]string{
		"--apply-signer-fd", "4", "--eval-signer-fd", "4",
	})
	if err == nil || !strings.Contains(err.Error(), "must be distinct") {
		t.Fatalf("expected aliased signer descriptor rejection, got %v", err)
	}
}

func TestCleanChildEnvironmentSetsTrustedRuntimeMarker(t *testing.T) {
	environment := cleanChildEnvironment(1234, 3, "/opt/axiom-verification", "/root", "")
	want := map[string]string{
		trustedRuntimeEnv:  "1",
		brokerActiveEnv:    "1",
		"PYTHONUNBUFFERED": "1",
	}
	for name, value := range want {
		expected := name + "=" + value
		found := false
		for _, entry := range environment {
			if entry == expected {
				found = true
				break
			}
		}
		if !found {
			t.Fatalf("cleaned child environment missing %q; got %v", expected, environment)
		}
	}
	// The marker must originate from the empty child environment, never inherited.
	for _, entry := range environment {
		if strings.HasPrefix(entry, trustedRuntimeEnv+"=") && entry != trustedRuntimeEnv+"=1" {
			t.Fatalf("trusted-runtime marker must be exactly 1, got %q", entry)
		}
	}
}

func TestCodexScratchPolicyRejectsRootOwnedHomeForNonRootRuntime(t *testing.T) {
	if err := validateCodexScratchPolicy(os.ModeDir|0700, 0, 1000); err == nil ||
		!strings.Contains(err.Error(), "not runtime-owned") {
		t.Fatalf("expected root-owned scratch rejection, got %v", err)
	}
}

func TestCodexScratchPolicyAcceptsOperatorOwnedProtectedHome(t *testing.T) {
	if err := validateCodexScratchPolicy(os.ModeDir|0700, 1000, 1000); err != nil {
		t.Fatalf("expected operator-owned 0700 scratch acceptance, got %v", err)
	}
}

func frozenTestInventory() map[string]any {
	return map[string]any{
		"schema": "axiom-encode/retired-apply-inventory/v1",
		"repositories": []any{map[string]any{
			"repository":    "TheAxiomFoundation/rulespec-us",
			"cutoff_commit": strings.Repeat("a", 40),
			"artifacts": []any{map[string]any{
				"path":   ".axiom/encoding-manifests/frozen.json",
				"sha256": strings.Repeat("b", 64),
				"covered_files": []any{map[string]any{
					"path": "statute/frozen.yaml", "sha256": strings.Repeat("c", 64),
				}},
			}},
		}},
	}
}

func retiredTestTrustRoots() map[string]any {
	payload := baseTrustRoots()
	payload["schema"] = "axiom-encode/signing-trust-roots/v4"
	payload["retired_apply_ed25519_public_keys"] = []string{encodedTestPublicKey(0xef)}
	payload["retired_apply_inventory"] = frozenTestInventory()
	return payload
}

func TestProtectedTrustRootsAcceptsFrozenRetiredApplyV4(t *testing.T) {
	for _, pluralCorpus := range []bool{false, true} {
		payload := retiredTestTrustRoots()
		if pluralCorpus {
			delete(payload, "corpus_release_ed25519_public_key")
			payload["corpus_release_ed25519_public_keys"] = []string{encodedTestPublicKey(0x17), encodedTestPublicKey(0x18)}
		}
		raw, _ := json.Marshal(payload)
		active, _, _, retired, inventory, err := parseProtectedTrustRoots(raw, "/protected/trust.json")
		if err != nil || len(retired) != 1 || retired[0][0] != 0xef || active[0] != 0xab || inventory.Repositories[0].CutoffCommit != strings.Repeat("a", 40) {
			t.Fatalf("frozen retired trust rejected or changed active key: %v", err)
		}
	}
}

func TestProtectedTrustRootsRejectsMalformedRetiredApplyV4(t *testing.T) {
	for _, testCase := range []struct {
		name   string
		mutate func(map[string]any)
	}{
		{"v3 retirement", func(p map[string]any) { p["schema"] = "axiom-encode/signing-trust-roots/v3" }},
		{"missing roots", func(p map[string]any) { delete(p, "retired_apply_ed25519_public_keys") }},
		{"missing inventory", func(p map[string]any) { delete(p, "retired_apply_inventory") }},
		{"null roots", func(p map[string]any) { p["retired_apply_ed25519_public_keys"] = nil }},
		{"empty roots", func(p map[string]any) { p["retired_apply_ed25519_public_keys"] = []string{} }},
		{"bad root", func(p map[string]any) { p["retired_apply_ed25519_public_keys"] = []string{"bad-base64!"} }},
		{"short root", func(p map[string]any) {
			p["retired_apply_ed25519_public_keys"] = []string{base64.StdEncoding.EncodeToString([]byte("short"))}
		}},
		{"duplicate root", func(p map[string]any) {
			p["retired_apply_ed25519_public_keys"] = []string{encodedTestPublicKey(0xef), encodedTestPublicKey(0xef)}
		}},
		{"active overlap", func(p map[string]any) { p["retired_apply_ed25519_public_keys"] = []string{encodedTestPublicKey(0xab)} }},
		{"eval overlap", func(p map[string]any) { p["retired_apply_ed25519_public_keys"] = []string{encodedTestPublicKey(0xcd)} }},
		{"corpus overlap", func(p map[string]any) { p["retired_apply_ed25519_public_keys"] = []string{encodedTestPublicKey(0x17)} }},
		{"null inventory", func(p map[string]any) { p["retired_apply_inventory"] = nil }},
		{"unknown inventory field", func(p map[string]any) { p["retired_apply_inventory"].(map[string]any)["caller_override"] = true }},
		{"bad inventory schema", func(p map[string]any) { p["retired_apply_inventory"].(map[string]any)["schema"] = "v0" }},
		{"duplicate repository", func(p map[string]any) {
			inv := p["retired_apply_inventory"].(map[string]any)
			repos := inv["repositories"].([]any)
			inv["repositories"] = append(repos, repos[0])
		}},
		{"bad cutoff", func(p map[string]any) {
			repo := p["retired_apply_inventory"].(map[string]any)["repositories"].([]any)[0].(map[string]any)
			repo["cutoff_commit"] = "main"
		}},
		{"uncanonical repository", func(p map[string]any) {
			repo := p["retired_apply_inventory"].(map[string]any)["repositories"].([]any)[0].(map[string]any)
			repo["repository"] = "https://github.com/TheAxiomFoundation/rulespec-us"
		}},
		{"duplicate artifact", func(p map[string]any) {
			repo := p["retired_apply_inventory"].(map[string]any)["repositories"].([]any)[0].(map[string]any)
			artifacts := repo["artifacts"].([]any)
			repo["artifacts"] = append(artifacts, artifacts[0])
		}},
		{"traversal artifact", func(p map[string]any) {
			artifact := p["retired_apply_inventory"].(map[string]any)["repositories"].([]any)[0].(map[string]any)["artifacts"].([]any)[0].(map[string]any)
			artifact["path"] = "../frozen.json"
		}},
		{"bad artifact digest", func(p map[string]any) {
			artifact := p["retired_apply_inventory"].(map[string]any)["repositories"].([]any)[0].(map[string]any)["artifacts"].([]any)[0].(map[string]any)
			artifact["sha256"] = "sha256:" + strings.Repeat("b", 64)
		}},
		{"missing coverage", func(p map[string]any) {
			artifact := p["retired_apply_inventory"].(map[string]any)["repositories"].([]any)[0].(map[string]any)["artifacts"].([]any)[0].(map[string]any)
			delete(artifact, "covered_files")
		}},
		{"duplicate covered path", func(p map[string]any) {
			artifact := p["retired_apply_inventory"].(map[string]any)["repositories"].([]any)[0].(map[string]any)["artifacts"].([]any)[0].(map[string]any)
			files := artifact["covered_files"].([]any)
			artifact["covered_files"] = append(files, files[0])
		}},
	} {
		t.Run(testCase.name, func(t *testing.T) {
			payload := retiredTestTrustRoots()
			testCase.mutate(payload)
			raw, _ := json.Marshal(payload)
			if _, _, _, _, _, err := parseProtectedTrustRoots(raw, "/protected/trust.json"); err == nil {
				t.Fatal("malformed retired trust was accepted")
			}
		})
	}
}

func TestProtectedTrustRootsRejectsDuplicateInventoryJSONFields(t *testing.T) {
	raw, _ := json.Marshal(retiredTestTrustRoots())
	raw = bytes.Replace(raw, []byte(`"cutoff_commit":`), []byte(`"cutoff_commit":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","cutoff_commit":`), 1)
	if _, _, _, _, _, err := parseProtectedTrustRoots(raw, "/protected/trust.json"); err == nil || !strings.Contains(err.Error(), "duplicate") {
		t.Fatalf("duplicate inventory field accepted: %v", err)
	}
}

func TestBrokerStatusWithoutRetirementHasUnchangedJSON(t *testing.T) {
	status := brokerStatus{Capabilities: []string{}, ApplyPublicKey: []byte{1}, EvalPublicKey: []byte{2}, CorpusReleasePublicKey: []byte{3}, CorpusReleasePublicKeys: [][]byte{{3}}}
	raw, err := json.Marshal(status)
	if err != nil {
		t.Fatal(err)
	}
	if string(raw) != `{"capabilities":[],"apply_public_key":"AQ==","eval_public_key":"Ag==","corpus_release_public_key":"Aw==","corpus_release_public_keys":["Aw=="]}` {
		t.Fatalf("legacy status bytes changed: %s", raw)
	}
}

func TestRetiredInventoryPathsPreserveLegacyPOSIXNames(t *testing.T) {
	for _, path := range []string{"us/statutes/26/26:1.yaml", "us/statutes/42/1437c–1.yaml", "He-W 704.yaml"} {
		if !validRetiredArtifactPath(path) {
			t.Fatalf("historical path rejected: %q", path)
		}
	}
	for _, path := range []string{"/absolute", "../traversal", "a//b", "a/./b", "a/../b", "-option", ":(glob)*", "a?b", "a[b", "a\\b", "a\x00b", "a\u0085b"} {
		if validRetiredArtifactPath(path) {
			t.Fatalf("ambiguous path accepted: %q", path)
		}
	}
}

func TestRetiredInventoryCoveredHistoricalVersions(t *testing.T) {
	payload := retiredTestTrustRoots()
	artifact := payload["retired_apply_inventory"].(map[string]any)["repositories"].([]any)[0].(map[string]any)["artifacts"].([]any)[0].(map[string]any)
	files := artifact["covered_files"].([]any)
	artifact["covered_files"] = append(files, map[string]any{"path": "statute/frozen.yaml", "sha256": strings.Repeat("d", 64), "source_commit": strings.Repeat("e", 40)})
	raw, _ := json.Marshal(payload)
	if _, _, _, _, _, err := parseProtectedTrustRoots(raw, "/protected/trust.json"); err != nil {
		t.Fatalf("distinct frozen historical version rejected: %v", err)
	}
	for _, source := range []any{nil, "main", strings.Repeat("e", 39)} {
		artifact["covered_files"].([]any)[1].(map[string]any)["source_commit"] = source
		raw, _ := json.Marshal(payload)
		if _, _, _, _, _, err := parseProtectedTrustRoots(raw, "/protected/trust.json"); err == nil {
			t.Fatalf("invalid historical source commit accepted: %v", source)
		}
	}
}

func TestLegacyTrustRootsRetainsSizeBound(t *testing.T) {
	legacy, _ := json.Marshal(baseTrustRoots())
	legacy = append(legacy, bytes.Repeat([]byte(" "), 64*1024)...)
	if _, _, _, _, _, err := parseProtectedTrustRoots(legacy, "/protected/trust.json"); err == nil || !strings.Contains(err.Error(), "64 KiB") {
		t.Fatalf("legacy size bound changed: %v", err)
	}
	retired, _ := json.Marshal(retiredTestTrustRoots())
	retired = append(retired, bytes.Repeat([]byte(" "), 64*1024)...)
	if _, _, _, _, _, err := parseProtectedTrustRoots(retired, "/protected/trust.json"); err != nil {
		t.Fatalf("v4 inventory cannot exceed legacy bound: %v", err)
	}
}

// This contract is also consumed by Python without requiring a Go toolchain.
type waiverAuditWorkersContract struct {
	Env       string   `json:"env"`
	MaxDigits int      `json:"max_digits"`
	MaxValue  int      `json:"max_value"`
	Accepted  []string `json:"accepted"`
	Refused   []string `json:"refused"`
}

func loadWaiverAuditWorkersContract(t testing.TB) waiverAuditWorkersContract {
	t.Helper()
	raw, err := os.ReadFile("testdata/waiver_audit_workers.json")
	if err != nil {
		t.Fatal(err)
	}
	var contract waiverAuditWorkersContract
	if err := json.Unmarshal(raw, &contract); err != nil {
		t.Fatal(err)
	}
	return contract
}

func TestCleanChildEnvironmentWithoutWaiverAuditWorkers(t *testing.T) {
	const name = "AXIOM_ENCODE_WAIVER_AUDIT_WORKERS"
	t.Setenv(name, "")
	if err := os.Unsetenv(name); err != nil {
		t.Fatal(err)
	}
	environment := cleanWaiverAuditWorkersTestEnvironment(t)
	for _, entry := range environment {
		if strings.HasPrefix(entry, name+"=") {
			t.Fatal("absent audit workers was added")
		}
	}
	// All fixed entries match the pre-forwarding environment. Existing named
	// parent passthroughs are checked elsewhere and need not expose their values.
	fixed := make(map[string]string)
	for _, entry := range environment {
		name, value, _ := strings.Cut(entry, "=")
		ambient := false
		for _, parentName := range parentOnlyEnvironmentNames {
			if name == parentName {
				ambient = true
				break
			}
		}
		if !ambient {
			fixed[name] = value
		}
	}
	want := map[string]string{
		"GIT_CONFIG_GLOBAL": "/dev/null", "GIT_CONFIG_NOSYSTEM": "1",
		"GIT_TERMINAL_PROMPT": "0", "HOME": "/trusted", "LANG": "C.UTF-8",
		"PATH": "/trusted/bin", "PYTHONNOUSERSITE": "1", "PYTHONSAFEPATH": "1",
		"PYTHONUNBUFFERED": "1", "PYTHONDONTWRITEBYTECODE": "1",
		"XDG_CONFIG_HOME": "/trusted/.empty-config", "XDG_DATA_HOME": "/trusted/.empty-data",
		brokerActiveEnv: "1", trustedRuntimeEnv: "1", brokerFDEnv: "11", brokerPIDEnv: "10",
	}
	if !reflect.DeepEqual(fixed, want) {
		t.Fatal("absent setting changed fixed child environment")
	}
}

func TestCleanChildEnvironmentForwardsWaiverAuditWorkers(t *testing.T) {
	contract := loadWaiverAuditWorkersContract(t)
	for _, raw := range contract.Accepted {
		t.Run(raw, func(t *testing.T) {
			t.Setenv(contract.Env, raw)
			environment := cleanWaiverAuditWorkersTestEnvironment(t)
			count := 0
			for _, entry := range environment {
				if strings.HasPrefix(entry, contract.Env+"=") {
					count++
					if entry != contract.Env+"="+raw {
						t.Fatal("audit workers changed during forwarding")
					}
				}
			}
			if count != 1 {
				t.Fatalf("expected one audit worker setting, got %d", count)
			}
		})
	}
}

// Exercise main's actual exit path in a subprocess. Invalid settings must win
// even over invalid command options: neither broker nor child can be started.
func TestCleanChildEnvironmentRefusesMalformedWaiverAuditWorkers(t *testing.T) {
	contract := loadWaiverAuditWorkersContract(t)
	for index, raw := range contract.Refused {
		t.Run(strconv.Itoa(index), func(t *testing.T) {
			t.Setenv(contract.Env, raw)
			t.Setenv("AXIOM_TEST_WAIVER_WORKERS_MAIN", "1")
			command := exec.Command(os.Args[0], "-test.run=^TestWaiverAuditWorkersSupervisorMain$")
			output, err := command.CombinedOutput()
			if err == nil {
				t.Fatal("supervisor accepted malformed audit workers")
			}
			exit, ok := err.(*exec.ExitError)
			if !ok || exit.ExitCode() != 2 {
				t.Fatalf("expected supervisor exit 2, got %v", err)
			}
			if !strings.Contains(string(output), "signing supervisor: "+contract.Env+" must be") {
				t.Fatalf("expected named audit worker rejection before runtime/broker/child setup, got %s", output)
			}
		})
	}
}

func TestWaiverAuditWorkersSupervisorMain(t *testing.T) {
	if os.Getenv("AXIOM_TEST_WAIVER_WORKERS_MAIN") != "1" {
		return
	}
	// These options cannot reach broker setup or child execution. Requiring the
	// worker diagnostic proves refusal precedes all supervised command paths.
	os.Args = []string{"axiom-encode-signing-supervisor", "--invalid-supervisor-option"}
	main()
	os.Exit(0)
}

// Exhaust the bounded decimal domain independently of the implementation.
func TestWaiverAuditWorkersContract(t *testing.T) {
	contract := loadWaiverAuditWorkersContract(t)
	if contract.Env != waiverAuditWorkersEnv || contract.MaxDigits != waiverAuditWorkersMaxDigits || contract.MaxValue != waiverAuditWorkersMax {
		t.Fatal("Go policy differs from the shared Python compatibility contract")
	}
	accepted := make(map[string]bool)
	for _, raw := range contract.Accepted {
		accepted[raw] = true
	}
	if len(accepted) != 43 {
		t.Fatal("incomplete shared accepted set")
	}
	for digits := 1; digits <= 2; digits++ {
		limit := 10
		if digits == 2 {
			limit = 100
		}
		for value := 0; value < limit; value++ {
			raw := fmt.Sprintf("%0*d", digits, value)
			if got := validateWaiverAuditWorkers(raw) == nil; got != accepted[raw] {
				t.Fatalf("contract mismatch for %q: accepted=%v", raw, got)
			}
		}
	}
}

func FuzzWaiverAuditWorkersEnvironment(f *testing.F) {
	contract := loadWaiverAuditWorkersContract(f)
	accepted := make(map[string]bool)
	for _, raw := range contract.Accepted {
		accepted[raw] = true
		f.Add(raw)
	}
	for _, raw := range contract.Refused {
		f.Add(raw)
	}
	f.Add("\x00")
	f.Add("٤")
	f.Fuzz(func(t *testing.T, raw string) {
		want := accepted[raw]
		if got := validateWaiverAuditWorkers(raw) == nil; got != want {
			t.Fatalf("accepted=%v, well-formed=%v for %q", got, want, raw)
		}
		// NUL cannot enter an OS environment, but still exercises the parser
		// above, so validation covers arbitrary strings without a skip.
		if strings.ContainsRune(raw, 0) {
			return
		}
		t.Setenv(contract.Env, raw)
		workers, err := waiverAuditWorkersFromEnvironment()
		if err != nil {
			if want || workers != "" {
				t.Fatal("refusal returned a forwardable setting")
			}
			return
		}
		if !want {
			t.Fatal("invalid setting was not refused")
		}
		environment := cleanChildEnvironment(10, 11, "/trusted/bin", "/trusted", workers)
		count := 0
		for _, entry := range environment {
			if strings.HasPrefix(entry, contract.Env+"=") {
				count++
				forwarded := strings.TrimPrefix(entry, contract.Env+"=")
				if forwarded != raw {
					t.Fatal("worker setting changed")
				}
				value, parseError := strconv.Atoi(forwarded)
				expected, _ := strconv.Atoi(raw)
				if parseError != nil || value != expected || value < 0 || value > contract.MaxValue {
					t.Fatal("forwarded integer meaning changed")
				}
				// The shared Python test executes the real parser for every
				// accepted spelling, including zero's serial-worker meaning.
			}
		}
		if count != 1 {
			t.Fatalf("expected one forwarded setting, got %d", count)
		}
	})
}

func cleanWaiverAuditWorkersTestEnvironment(t testing.TB) []string {
	t.Helper()
	workers, err := waiverAuditWorkersFromEnvironment()
	if err != nil {
		t.Fatal(err)
	}
	return cleanChildEnvironment(10, 11, "/trusted/bin", "/trusted", workers)
}
