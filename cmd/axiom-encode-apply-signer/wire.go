//go:build darwin || linux

package main

import (
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"runtime"
)

// External signer protocol v2. These constants mirror the compiled trust
// boundary in cmd/axiom-encode-signing-supervisor (the broker is the protocol
// v2 *client*; this binary is the *server*). They are a stable wire contract:
// the supervisor's broker signs the challenge domain over the returned public
// key and verifies every persisted signature against the signature domain, so a
// drift here fails closed at the broker rather than producing a bad signature.
// The end-to-end tests exercise the real supervisor binary, which structurally
// guarantees these stay in lockstep.
const (
	signerProtocolVersion = 2
	maxFrameBytes         = 64 * 1024 * 1024
	maxJSONDepth          = 32
	maxJSONKeyBytes       = 256

	// signerChallengeDomain and signerSignatureDomain bind every signature to a
	// distinct operation. The trailing NUL is part of the domain.
	signerChallengeDomain = "axiom-encode/external-signer-challenge/v2\x00"
	signerSignatureDomain = "axiom-encode/external-signer-sign/v2\x00"
)

// scopeApply and scopeEval are the only scopes protocol v2 recognizes. Apply
// signatures cannot verify as eval signatures or vice versa because the scope is
// mixed into every signed message.
const (
	scopeApply = "apply_ed25519"
	scopeEval  = "eval_ed25519"
)

func isKnownScope(scope string) bool {
	return scope == scopeApply || scope == scopeEval
}

// signerRequest is the exact inbound shape the broker sends. A challenge request
// carries {version,id,op,scope,challenge}; a sign request carries
// {version,id,op,scope,payload}. Field presence is tracked so an omitted field
// is distinguishable from an explicitly empty one, which a capability protocol
// requires.
type signerRequest struct {
	Version       int    `json:"version"`
	ID            int64  `json:"id"`
	Operation     string `json:"op"`
	Scope         string `json:"scope"`
	Challenge     []byte `json:"challenge,omitempty"`
	Payload       []byte `json:"payload,omitempty"`
	presentFields map[string]struct{}
}

var signerRequestFields = map[string]struct{}{
	"version":   {},
	"id":        {},
	"op":        {},
	"scope":     {},
	"challenge": {},
	"payload":   {},
}

func (request *signerRequest) UnmarshalJSON(raw []byte) error {
	var encodedFields map[string]json.RawMessage
	if err := json.Unmarshal(raw, &encodedFields); err != nil {
		return err
	}
	if encodedFields == nil {
		return errors.New("external signer request must be a JSON object")
	}
	defer func() {
		for _, encoded := range encodedFields {
			zero(encoded)
		}
	}()
	for name := range encodedFields {
		if _, allowed := signerRequestFields[name]; !allowed {
			return fmt.Errorf("json: unknown field %q", name)
		}
	}
	type wireSignerRequest signerRequest
	var decoded wireSignerRequest
	if err := json.Unmarshal(raw, &decoded); err != nil {
		return err
	}
	*request = signerRequest(decoded)
	request.presentFields = make(map[string]struct{}, len(encodedFields))
	for name := range encodedFields {
		request.presentFields[name] = struct{}{}
	}
	return nil
}

func (request *signerRequest) hasField(name string) bool {
	_, present := request.presentFields[name]
	return present
}

func (request *signerRequest) hasExactFields(expected ...string) bool {
	if len(request.presentFields) != len(expected) {
		return false
	}
	for _, name := range expected {
		if _, present := request.presentFields[name]; !present {
			return false
		}
	}
	return true
}

// challengeResponse, signResponse, and errorResponse are marshaled with exactly
// the fields the broker's strict decoder admits. The broker rejects any extra,
// missing, or duplicate field, so these types must not gain omitempty fields.
type challengeResponse struct {
	Version   int    `json:"version"`
	ID        int64  `json:"id"`
	OK        bool   `json:"ok"`
	PublicKey []byte `json:"public_key"`
	Signature []byte `json:"signature"`
}

type signResponse struct {
	Version   int    `json:"version"`
	ID        int64  `json:"id"`
	OK        bool   `json:"ok"`
	Signature []byte `json:"signature"`
}

type errorResponse struct {
	Version int    `json:"version"`
	ID      int64  `json:"id"`
	OK      bool   `json:"ok"`
	Error   string `json:"error"`
}

func sendFrame(writer io.Writer, payload any) error {
	raw, err := json.Marshal(payload)
	if err != nil {
		return err
	}
	defer zero(raw)
	if len(raw) > maxFrameBytes {
		return errors.New("external signer frame exceeds its size limit")
	}
	header := make([]byte, 4)
	binary.BigEndian.PutUint32(header, uint32(len(raw)))
	if err := writeAll(writer, header); err != nil {
		return err
	}
	return writeAll(writer, raw)
}

func receiveFrame(reader io.Reader, destination *signerRequest) error {
	header := make([]byte, 4)
	if _, err := io.ReadFull(reader, header); err != nil {
		return err
	}
	length := binary.BigEndian.Uint32(header)
	if length > maxFrameBytes {
		return errors.New("external signer frame exceeds its size limit")
	}
	raw := make([]byte, int(length))
	defer zero(raw)
	if _, err := io.ReadFull(reader, raw); err != nil {
		return err
	}
	if err := rejectDuplicateJSONKeys(raw); err != nil {
		return fmt.Errorf("external signer frame is malformed: %w", err)
	}
	// signerRequest owns its strict unknown-field and field-presence decode.
	// Decode directly from the frame so no second, unzeroizable copy of the
	// signing payload is retained in a decoder read buffer.
	if err := json.Unmarshal(raw, destination); err != nil {
		return fmt.Errorf("external signer frame is malformed: %w", err)
	}
	return nil
}

func writeAll(writer io.Writer, data []byte) error {
	for len(data) != 0 {
		written, err := writer.Write(data)
		if err != nil {
			return err
		}
		if written == 0 {
			return io.ErrShortWrite
		}
		data = data[written:]
	}
	return nil
}

func zero(value []byte) {
	for index := range value {
		value[index] = 0
	}
	runtime.KeepAlive(value)
}

// rejectDuplicateJSONKeys mirrors the supervisor's frame validation: it rejects
// duplicate object keys, trailing data, oversized keys, and excessive nesting
// without materializing signing payloads as immutable Go strings.
func rejectDuplicateJSONKeys(raw []byte) error {
	if !json.Valid(raw) {
		return errors.New("frame is not valid JSON or contains trailing JSON")
	}
	scanner := jsonKeyScanner{raw: raw}
	if err := scanner.scanValue(0); err != nil {
		return err
	}
	scanner.skipWhitespace()
	if scanner.offset != len(raw) {
		return errors.New("frame contains trailing JSON")
	}
	return nil
}

type jsonKeyScanner struct {
	raw    []byte
	offset int
}

func (scanner *jsonKeyScanner) skipWhitespace() {
	for scanner.offset < len(scanner.raw) {
		switch scanner.raw[scanner.offset] {
		case ' ', '\t', '\n', '\r':
			scanner.offset++
		default:
			return
		}
	}
}

func (scanner *jsonKeyScanner) scanValue(depth int) error {
	if depth > maxJSONDepth {
		return fmt.Errorf("JSON nesting exceeds the %d-level limit", maxJSONDepth)
	}
	scanner.skipWhitespace()
	if scanner.offset >= len(scanner.raw) {
		return io.ErrUnexpectedEOF
	}
	switch scanner.raw[scanner.offset] {
	case '{':
		return scanner.scanObject(depth)
	case '[':
		return scanner.scanArray(depth)
	case '"':
		_, err := scanner.scanString()
		return err
	default:
		start := scanner.offset
		for scanner.offset < len(scanner.raw) {
			switch scanner.raw[scanner.offset] {
			case ' ', '\t', '\n', '\r', ',', ']', '}':
				if scanner.offset == start {
					return errors.New("JSON value is malformed")
				}
				return nil
			default:
				scanner.offset++
			}
		}
		if scanner.offset == start {
			return errors.New("JSON value is malformed")
		}
		return nil
	}
}

func (scanner *jsonKeyScanner) scanObject(depth int) error {
	scanner.offset++
	scanner.skipWhitespace()
	if scanner.raw[scanner.offset] == '}' {
		scanner.offset++
		return nil
	}
	seen := make(map[string]struct{})
	for {
		scanner.skipWhitespace()
		if scanner.offset >= len(scanner.raw) || scanner.raw[scanner.offset] != '"' {
			return errors.New("object key is not a string")
		}
		encodedKey, err := scanner.scanString()
		if err != nil {
			return err
		}
		if len(encodedKey) > maxJSONKeyBytes {
			return fmt.Errorf("object key exceeds %d bytes", maxJSONKeyBytes)
		}
		var key string
		if err := json.Unmarshal(encodedKey, &key); err != nil {
			return err
		}
		if _, duplicate := seen[key]; duplicate {
			return fmt.Errorf("duplicate object key %q", key)
		}
		seen[key] = struct{}{}
		scanner.skipWhitespace()
		if scanner.raw[scanner.offset] != ':' {
			return errors.New("object key is missing its value separator")
		}
		scanner.offset++
		if err := scanner.scanValue(depth + 1); err != nil {
			return err
		}
		scanner.skipWhitespace()
		switch scanner.raw[scanner.offset] {
		case '}':
			scanner.offset++
			return nil
		case ',':
			scanner.offset++
		default:
			return errors.New("object is not terminated")
		}
	}
}

func (scanner *jsonKeyScanner) scanArray(depth int) error {
	scanner.offset++
	scanner.skipWhitespace()
	if scanner.raw[scanner.offset] == ']' {
		scanner.offset++
		return nil
	}
	for {
		if err := scanner.scanValue(depth + 1); err != nil {
			return err
		}
		scanner.skipWhitespace()
		switch scanner.raw[scanner.offset] {
		case ']':
			scanner.offset++
			return nil
		case ',':
			scanner.offset++
		default:
			return errors.New("array is not terminated")
		}
	}
}

func (scanner *jsonKeyScanner) scanString() ([]byte, error) {
	start := scanner.offset
	scanner.offset++
	for scanner.offset < len(scanner.raw) {
		switch scanner.raw[scanner.offset] {
		case '"':
			scanner.offset++
			return scanner.raw[start:scanner.offset], nil
		case '\\':
			scanner.offset += 2
		default:
			scanner.offset++
		}
	}
	return nil, errors.New("JSON string is not terminated")
}
