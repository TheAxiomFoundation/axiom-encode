//go:build darwin || linux

package main

import (
	"bytes"
	"encoding/binary"
	"os"
	"strings"
	"testing"
)

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
	environment := cleanChildEnvironment(10, 11, "/trusted/bin", "/trusted")
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
		`{"version":4,"id":0,"op":"initialize","apply_public_key":"q6urq6urq6urq6urq6urq6urq6urq6urq6urq6urq6s=","eval_public_key":"zc3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc3Nzc0=","corpus_release_public_key":"FxcXFxcXFxcXFxcXFxcXFxcXFxcXFxcXFxcXFxcXFxc="}`,
	)
	if !valid.hasValidInitializationShape() {
		t.Fatal("expected three distinct initialization roots to be valid")
	}

	aliased := valid
	aliased.CorpusReleasePublicKey = append([]byte(nil), valid.ApplyPublicKey...)
	if aliased.hasValidInitializationShape() {
		t.Fatal("expected aliased corpus release root to be rejected")
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
	environment := cleanChildEnvironment(1234, 3, "/opt/axiom-verification", "/root")
	want := map[string]string{
		trustedRuntimeEnv: "1",
		brokerActiveEnv:   "1",
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
