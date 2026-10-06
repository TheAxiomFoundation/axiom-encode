//go:build darwin || linux

package main

import (
	"bytes"
	"crypto/ed25519"
	"crypto/rand"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"net"
	"strings"
	"testing"
	"time"
)

// brokerEmulator drives a signerServer over a connection exactly as the
// compiled supervisor's broker does: length-prefixed protocol v2 JSON frames.
type brokerEmulator struct {
	t    *testing.T
	conn net.Conn
}

func (broker *brokerEmulator) send(request map[string]any) {
	broker.t.Helper()
	raw, err := json.Marshal(request)
	if err != nil {
		broker.t.Fatalf("marshal request: %v", err)
	}
	header := make([]byte, 4)
	binary.BigEndian.PutUint32(header, uint32(len(raw)))
	if _, err := broker.conn.Write(header); err != nil {
		broker.t.Fatalf("write header: %v", err)
	}
	if _, err := broker.conn.Write(raw); err != nil {
		broker.t.Fatalf("write body: %v", err)
	}
}

func (broker *brokerEmulator) receive() (map[string]json.RawMessage, []byte) {
	broker.t.Helper()
	_ = broker.conn.SetReadDeadline(time.Now().Add(5 * time.Second))
	header := make([]byte, 4)
	if _, err := readFull(broker.conn, header); err != nil {
		broker.t.Fatalf("read response header: %v", err)
	}
	length := binary.BigEndian.Uint32(header)
	body := make([]byte, length)
	if _, err := readFull(broker.conn, body); err != nil {
		broker.t.Fatalf("read response body: %v", err)
	}
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(body, &fields); err != nil {
		broker.t.Fatalf("unmarshal response: %v", err)
	}
	return fields, body
}

func readFull(reader net.Conn, buffer []byte) (int, error) {
	total := 0
	for total < len(buffer) {
		n, err := reader.Read(buffer[total:])
		total += n
		if err != nil {
			return total, err
		}
	}
	return total, nil
}

func decodeBytesField(t *testing.T, fields map[string]json.RawMessage, name string) []byte {
	t.Helper()
	raw, ok := fields[name]
	if !ok {
		t.Fatalf("response missing field %q", name)
	}
	var encoded string
	if err := json.Unmarshal(raw, &encoded); err != nil {
		t.Fatalf("field %q is not a JSON string: %v", name, err)
	}
	decoded, err := base64.StdEncoding.DecodeString(encoded)
	if err != nil {
		t.Fatalf("field %q is not base64: %v", name, err)
	}
	return decoded
}

func exactResponseKeys(fields map[string]json.RawMessage) []string {
	keys := make([]string, 0, len(fields))
	for name := range fields {
		keys = append(keys, name)
	}
	return keys
}

// withServer runs a signerServer over an in-process pipe and gives the test a
// broker emulator plus the server's public key and audit buffer.
func withServer(t *testing.T, scope string, fn func(broker *brokerEmulator, publicKey ed25519.PublicKey, audit *bytes.Buffer)) {
	t.Helper()
	public, private, err := ed25519.GenerateKey(rand.Reader)
	if err != nil {
		t.Fatalf("generate key: %v", err)
	}
	serverConn, clientConn := net.Pipe()
	audit := &bytes.Buffer{}
	server := &signerServer{
		scope:      scope,
		privateKey: private,
		publicKey:  append([]byte(nil), public...),
		audit:      &auditLog{writer: audit},
	}
	done := make(chan error, 1)
	go func() { done <- server.serveConnection(serverConn) }()

	broker := &brokerEmulator{t: t, conn: clientConn}
	fn(broker, public, audit)

	_ = clientConn.Close()
	select {
	case <-done:
	case <-time.After(5 * time.Second):
		t.Fatal("server did not stop after connection close")
	}

	// Global invariant: no private-key material ever reaches the audit stream.
	auditText := audit.String()
	for _, secret := range []string{
		base64.StdEncoding.EncodeToString(private.Seed()),
		base64.StdEncoding.EncodeToString(private),
	} {
		if strings.Contains(auditText, secret) {
			t.Fatal("audit stream leaked private key material")
		}
	}
}

func TestChallengeThenSignProducesDomainBoundSignatures(t *testing.T) {
	withServer(t, scopeApply, func(broker *brokerEmulator, publicKey ed25519.PublicKey, audit *bytes.Buffer) {
		nonce := make([]byte, 32)
		if _, err := rand.Read(nonce); err != nil {
			t.Fatal(err)
		}
		broker.send(map[string]any{
			"version": 2, "id": 1, "op": "challenge",
			"scope": scopeApply, "challenge": nonce,
		})
		fields, _ := broker.receive()
		if got := exactResponseKeys(fields); len(got) != 5 {
			t.Fatalf("challenge response must have exactly 5 fields, got %v", got)
		}
		gotPublic := decodeBytesField(t, fields, "public_key")
		if !bytes.Equal(gotPublic, publicKey) {
			t.Fatal("challenge response returned the wrong public key")
		}
		challengeSig := decodeBytesField(t, fields, "signature")
		challengeMessage := []byte(signerChallengeDomain + scopeApply + "\x00")
		challengeMessage = append(challengeMessage, nonce...)
		if !ed25519.Verify(publicKey, challengeMessage, challengeSig) {
			t.Fatal("challenge signature does not verify over the challenge domain")
		}

		payload := []byte(`{"citation":"1 USC 1","schema_version":"axiom-encode/applied-rulespec/v5"}`)
		broker.send(map[string]any{
			"version": 2, "id": 2, "op": "sign",
			"scope": scopeApply, "payload": payload,
		})
		signFields, _ := broker.receive()
		if got := exactResponseKeys(signFields); len(got) != 4 {
			t.Fatalf("sign response must have exactly 4 fields, got %v", got)
		}
		signSig := decodeBytesField(t, signFields, "signature")
		signMessage := []byte(signerSignatureDomain + scopeApply + "\x00")
		signMessage = append(signMessage, payload...)
		if !ed25519.Verify(publicKey, signMessage, signSig) {
			t.Fatal("sign signature does not verify over the signature domain")
		}
		// Domain separation: the apply signature must not verify as an eval one.
		evalMessage := []byte(signerSignatureDomain + scopeEval + "\x00")
		evalMessage = append(evalMessage, payload...)
		if ed25519.Verify(publicKey, evalMessage, signSig) {
			t.Fatal("apply signature must not verify under the eval scope")
		}

		auditText := audit.String()
		if !strings.Contains(auditText, "event=sign") || !strings.Contains(auditText, "payload_sha256=") {
			t.Fatalf("audit log missing sign record: %q", auditText)
		}
		if !strings.Contains(auditText, "citation=1 USC 1") {
			t.Fatalf("audit log missing sanitized citation: %q", auditText)
		}
	})
}

func TestSignedV5ManifestPayloadVerifiesWithPublicHalf(t *testing.T) {
	withServer(t, scopeApply, func(broker *brokerEmulator, publicKey ed25519.PublicKey, _ *bytes.Buffer) {
		// The exact canonical unsigned bytes the CLI presents to the broker:
		// applied-rulespec/v5 payload minus signature, sorted keys, compact.
		manifest := map[string]any{
			"schema_version": "axiom-encode/applied-rulespec/v5",
			"citation":       "1 USC 1",
			"applied_files": []any{
				map[string]any{"path": "us/statutes/1.yaml", "sha256": strings.Repeat("a", 64)},
			},
		}
		payload, err := json.Marshal(manifest)
		if err != nil {
			t.Fatal(err)
		}
		broker.send(map[string]any{
			"version": 2, "id": 1, "op": "sign",
			"scope": scopeApply, "payload": payload,
		})
		fields, _ := broker.receive()
		signature := decodeBytesField(t, fields, "signature")
		message := append([]byte(signerSignatureDomain+scopeApply+"\x00"), payload...)
		if !ed25519.Verify(publicKey, message, signature) {
			t.Fatal("v5 manifest signature does not verify with the apply public half")
		}
	})
}

func TestSignRefusesForeignScope(t *testing.T) {
	withServer(t, scopeApply, func(broker *brokerEmulator, _ ed25519.PublicKey, _ *bytes.Buffer) {
		broker.send(map[string]any{
			"version": 2, "id": 1, "op": "sign",
			"scope": scopeEval, "payload": []byte("x"),
		})
		fields, _ := broker.receive()
		assertRefused(t, fields, "only serves scope")
	})
}

func TestServerRejectsUnknownOperation(t *testing.T) {
	withServer(t, scopeApply, func(broker *brokerEmulator, _ ed25519.PublicKey, _ *bytes.Buffer) {
		broker.send(map[string]any{
			"version": 2, "id": 1, "op": "export_key", "scope": scopeApply,
		})
		fields, _ := broker.receive()
		assertRefused(t, fields, "not supported")
	})
}

func TestServerRejectsWrongProtocolVersion(t *testing.T) {
	withServer(t, scopeApply, func(broker *brokerEmulator, _ ed25519.PublicKey, _ *bytes.Buffer) {
		broker.send(map[string]any{
			"version": 1, "id": 1, "op": "challenge",
			"scope": scopeApply, "challenge": []byte("nonce"),
		})
		fields, _ := broker.receive()
		assertRefused(t, fields, "malformed")
	})
}

func TestServerRejectsNonIncreasingRequestID(t *testing.T) {
	withServer(t, scopeApply, func(broker *brokerEmulator, _ ed25519.PublicKey, _ *bytes.Buffer) {
		nonce := make([]byte, 32)
		_, _ = rand.Read(nonce)
		broker.send(map[string]any{
			"version": 2, "id": 5, "op": "challenge",
			"scope": scopeApply, "challenge": nonce,
		})
		if fields, _ := broker.receive(); !responseOK(fields) {
			t.Fatal("first challenge should succeed")
		}
		broker.send(map[string]any{
			"version": 2, "id": 5, "op": "sign",
			"scope": scopeApply, "payload": []byte("x"),
		})
		fields, _ := broker.receive()
		assertRefused(t, fields, "strictly increasing")
	})
}

func TestChallengeRejectsExtraField(t *testing.T) {
	withServer(t, scopeApply, func(broker *brokerEmulator, _ ed25519.PublicKey, _ *bytes.Buffer) {
		nonce := make([]byte, 32)
		_, _ = rand.Read(nonce)
		broker.send(map[string]any{
			"version": 2, "id": 1, "op": "challenge",
			"scope": scopeApply, "challenge": nonce, "payload": []byte("x"),
		})
		fields, _ := broker.receive()
		assertRefused(t, fields, "malformed")
	})
}

func TestServerRejectsDuplicateKeyFrame(t *testing.T) {
	withServer(t, scopeApply, func(broker *brokerEmulator, _ ed25519.PublicKey, _ *bytes.Buffer) {
		raw := []byte(`{"version":2,"id":1,"id":2,"op":"challenge","scope":"apply_ed25519","challenge":""}`)
		header := make([]byte, 4)
		binary.BigEndian.PutUint32(header, uint32(len(raw)))
		if _, err := broker.conn.Write(append(header, raw...)); err != nil {
			t.Fatalf("write duplicate-key frame: %v", err)
		}
		// A frame the server cannot parse to an ID drops the connection; the
		// emulator observes EOF on the next read.
		_ = broker.conn.SetReadDeadline(time.Now().Add(5 * time.Second))
		if _, err := readFull(broker.conn, make([]byte, 4)); err == nil {
			t.Fatal("expected connection close after duplicate-key frame")
		}
	})
}

func assertRefused(t *testing.T, fields map[string]json.RawMessage, substring string) {
	t.Helper()
	if responseOK(fields) {
		t.Fatalf("expected refusal containing %q, got ok response %v", substring, fields)
	}
	keys := exactResponseKeys(fields)
	if len(keys) != 4 {
		t.Fatalf("error response must have exactly 4 fields, got %v", keys)
	}
	var message string
	if err := json.Unmarshal(fields["error"], &message); err != nil {
		t.Fatalf("error field is not a string: %v", err)
	}
	if !strings.Contains(message, substring) {
		t.Fatalf("expected error containing %q, got %q", substring, message)
	}
}

func responseOK(fields map[string]json.RawMessage) bool {
	raw, ok := fields["ok"]
	if !ok {
		return false
	}
	var value bool
	if err := json.Unmarshal(raw, &value); err != nil {
		return false
	}
	return value
}
