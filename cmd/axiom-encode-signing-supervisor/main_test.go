//go:build darwin || linux

package main

import (
	"bytes"
	"encoding/binary"
	"strings"
	"testing"
)

func framed(raw string) []byte {
	header := make([]byte, 4)
	binary.BigEndian.PutUint32(header, uint32(len(raw)))
	return append(header, []byte(raw)...)
}

func TestReceiveFrameRejectsTrailingJSONValue(t *testing.T) {
	var request brokerRequest
	err := receiveFrame(
		bytes.NewReader(framed(`{"version":3,"id":1,"op":"status"} {}`)),
		&request,
	)
	if err == nil || !strings.Contains(err.Error(), "trailing JSON") {
		t.Fatalf("expected trailing JSON rejection, got %v", err)
	}
}

func TestReceiveFrameRejectsUnknownField(t *testing.T) {
	var request brokerRequest
	err := receiveFrame(
		bytes.NewReader(framed(`{"version":3,"id":1,"op":"status","legacy_key":"no"}`)),
		&request,
	)
	if err == nil || !strings.Contains(err.Error(), "unknown field") {
		t.Fatalf("expected unknown field rejection, got %v", err)
	}
}

func TestReceiveFrameRejectsDuplicateKeysAtEveryDepth(t *testing.T) {
	for _, raw := range []string{
		`{"version":3,"id":1,"id":2,"op":"status"}`,
		`{"version":3,"\u0069d":1,"id":2,"op":"status"}`,
		`{"version":3,"id":1,"op":"status","payload":{"x":1,"x":2}}`,
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
	raw := `{"version":3,"id":1,"op":"status","payload":` +
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
			raw:   `{"version":3,"id":1,"op":"status","payload":""}`,
			error: "status request is malformed",
		},
		{
			name:  "null payload on sign",
			raw:   `{"version":3,"id":1,"op":"apply_ed25519_sign","payload":null}`,
			error: "signing request is malformed",
		},
		{
			name:  "missing payload on sign",
			raw:   `{"version":3,"id":1,"op":"apply_ed25519_sign"}`,
			error: "signing request is malformed",
		},
		{
			name:  "empty forbidden initialization field",
			raw:   `{"version":3,"id":1,"op":"status","apply_public_key":""}`,
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
		`{"version":3,"id":1,"op":"apply_ed25519_sign","payload":""}`,
	)
	if err := validateBrokerRequestOperation(&request); err != nil {
		t.Fatalf("explicit empty signing payload should be valid: %v", err)
	}
}

func TestBrokerRequestIDsArePositiveAndStrictlyIncreasing(t *testing.T) {
	lastRequestID := int64(0)
	first := decodedRequest(t, `{"version":3,"id":1,"op":"status"}`)
	if err := validateBrokerRequestEnvelope(&first, &lastRequestID); err != nil {
		t.Fatalf("first request should be valid: %v", err)
	}
	for _, raw := range []string{
		`{"version":3,"id":1,"op":"status"}`,
		`{"version":3,"id":0,"op":"status"}`,
		`{"version":3,"id":-1,"op":"status"}`,
	} {
		request := decodedRequest(t, raw)
		err := validateBrokerRequestEnvelope(&request, &lastRequestID)
		if err == nil || !strings.Contains(err.Error(), "strictly increasing") {
			t.Fatalf("expected request ID rejection for %s, got %v", raw, err)
		}
	}
	third := decodedRequest(t, `{"version":3,"id":3,"op":"status"}`)
	if err := validateBrokerRequestEnvelope(&third, &lastRequestID); err != nil {
		t.Fatalf("later increasing request should be valid: %v", err)
	}
}

func TestBrokerRejectsRemovedV2Protocol(t *testing.T) {
	request := decodedRequest(t, `{"version":2,"id":1,"op":"status"}`)
	lastRequestID := int64(0)
	err := validateBrokerRequestEnvelope(&request, &lastRequestID)
	if err == nil || !strings.Contains(err.Error(), "malformed") {
		t.Fatalf("expected removed protocol rejection, got %v", err)
	}
}

func TestBrokerInitializationRejectsEmptyOrExtraFields(t *testing.T) {
	for _, raw := range []string{
		`{"version":3,"id":0,"op":"initialize","apply_public_key":""}`,
		`{"version":3,"id":0,"op":"initialize","apply_public_key":"q6urq6urq6urq6urq6urq6urq6urq6urq6urq6urq6s=","payload":""}`,
		`{"version":3,"id":0,"op":"initialize"}`,
	} {
		request := decodedRequest(t, raw)
		if request.hasValidInitializationShape() {
			t.Fatalf("expected initialization shape rejection for %s", raw)
		}
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
