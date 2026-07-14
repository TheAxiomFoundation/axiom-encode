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

func TestReceiveFrameRejectsUnknownField(t *testing.T) {
	var request signerRequest
	err := receiveFrame(bytes.NewReader(framed(
		`{"version":2,"id":1,"op":"challenge","scope":"apply_ed25519","legacy":"x"}`,
	)), &request)
	if err == nil || !strings.Contains(err.Error(), "unknown field") {
		t.Fatalf("expected unknown field rejection, got %v", err)
	}
}

func TestReceiveFrameRejectsDuplicateKeys(t *testing.T) {
	for _, raw := range []string{
		`{"version":2,"id":1,"id":2,"op":"sign","scope":"apply_ed25519","payload":""}`,
		`{"version":2,"id":1,"id":2,"op":"sign","scope":"apply_ed25519","payload":""}`,
	} {
		var request signerRequest
		err := receiveFrame(bytes.NewReader(framed(raw)), &request)
		if err == nil || !strings.Contains(err.Error(), "duplicate object key") {
			t.Fatalf("expected duplicate-key rejection for %s, got %v", raw, err)
		}
	}
}

func TestReceiveFrameRejectsTrailingJSON(t *testing.T) {
	var request signerRequest
	err := receiveFrame(bytes.NewReader(framed(
		`{"version":2,"id":1,"op":"sign","scope":"apply_ed25519","payload":""} {}`,
	)), &request)
	if err == nil || !strings.Contains(err.Error(), "trailing JSON") {
		t.Fatalf("expected trailing JSON rejection, got %v", err)
	}
}

func TestReceiveFrameRejectsOversizedKey(t *testing.T) {
	raw := `{"` + strings.Repeat("x", maxJSONKeyBytes+1) + `":1}`
	var request signerRequest
	err := receiveFrame(bytes.NewReader(framed(raw)), &request)
	if err == nil || !strings.Contains(err.Error(), "object key exceeds") {
		t.Fatalf("expected oversized-key rejection, got %v", err)
	}
}

func TestReceiveFrameRejectsExcessiveNesting(t *testing.T) {
	raw := `{"version":2,"id":1,"op":"sign","scope":"apply_ed25519","payload":` +
		strings.Repeat("[", maxJSONDepth+2) + `null` +
		strings.Repeat("]", maxJSONDepth+2) + `}`
	var request signerRequest
	err := receiveFrame(bytes.NewReader(framed(raw)), &request)
	if err == nil || !strings.Contains(err.Error(), "nesting exceeds") {
		t.Fatalf("expected excessive-nesting rejection, got %v", err)
	}
}

func TestReceiveFrameParsesExactChallengeFields(t *testing.T) {
	var request signerRequest
	if err := receiveFrame(bytes.NewReader(framed(
		`{"version":2,"id":7,"op":"challenge","scope":"apply_ed25519","challenge":"YQ=="}`,
	)), &request); err != nil {
		t.Fatalf("valid challenge should decode: %v", err)
	}
	if !request.hasExactFields("version", "id", "op", "scope", "challenge") {
		t.Fatalf("challenge fields not tracked exactly: %v", request.presentFields)
	}
	if request.ID != 7 || request.Operation != "challenge" || request.Scope != scopeApply {
		t.Fatalf("challenge decoded incorrectly: %+v", request)
	}
	if !bytes.Equal(request.Challenge, []byte("a")) {
		t.Fatalf("challenge nonce decoded incorrectly: %q", request.Challenge)
	}
}
