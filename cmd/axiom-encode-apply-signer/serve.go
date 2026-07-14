//go:build darwin || linux

package main

import (
	"crypto/ed25519"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"os"
	"syscall"
	"time"
)

// serveOptions configures one external-signer process.
type serveOptions struct {
	scope    string
	socketFD int
	keyFD    int
	readyFD  int
	binding  contextBinding
}

// signerServer holds one operation-scoped Ed25519 key and answers protocol v2
// requests. It signs only its provisioned scope and only the two domain-bound
// message shapes protocol v2 defines; there is no generic blob-signing path.
type signerServer struct {
	scope      string
	privateKey ed25519.PrivateKey
	publicKey  []byte
	lastID     int64
	audit      *auditLog
	signed     int
}

// runServe is the production entrypoint: bind the CI context, harden the
// process, ingest the key, then serve the pre-connected socket. The private key
// is zeroized before return.
func runServe(options serveOptions, environment func(string) string, auditWriter io.Writer) error {
	// The process is already hardened in main() before any argument parsing, so
	// no key material is ever resident while the process is dumpable.
	if !isKnownScope(options.scope) {
		return fmt.Errorf("--scope must be one of %q or %q", scopeApply, scopeEval)
	}
	info, err := options.binding.validate(environment)
	if err != nil {
		return err
	}

	var privateKey ed25519.PrivateKey
	if info.local {
		privateKey, err = generateEphemeralKey()
		if err != nil {
			return err
		}
		// Surface the throwaway public key so a developer can build a matching
		// trust root. This is a human-facing notice, kept off the audit stream.
		fmt.Fprintf(os.Stderr,
			"axiom-apply-signer local-dev throwaway public key (base64): %s\n",
			base64.StdEncoding.EncodeToString(rawPublicKey(privateKey)),
		)
	} else {
		if options.keyFD < 0 {
			return errors.New("--key-fd is required in CI mode")
		}
		// Signal readiness only now — after hardening (done in main) and context
		// validation, and BEFORE reading the key. The launcher blocks on this
		// signal and delivers the key over the pipe only once this process is
		// non-dumpable, closing the /proc/<pid>/fd race on the key descriptor.
		if err := signalReady(options.readyFD); err != nil {
			return err
		}
		privateKey, err = ingestPrivateKeyFromDescriptor(options.keyFD)
		if err != nil {
			return err
		}
	}
	defer zero(privateKey)

	connection, err := attachSignerSocket(options.socketFD)
	if err != nil {
		return err
	}
	defer connection.Close()

	audit := &auditLog{writer: auditWriter}
	server := &signerServer{
		scope:      options.scope,
		privateKey: privateKey,
		publicKey:  rawPublicKey(privateKey),
		audit:      audit,
	}
	audit.bind(server.scope, supervisorPublicKind(info), info)

	err = server.serveConnection(connection)
	audit.shutdown(server.scope, server.signed)
	return err
}

// signalReady writes one byte to the readiness descriptor to tell the launcher
// the process is hardened and about to read the key. A negative descriptor means
// no readiness gate was requested (e.g. a non-adversarial test harness that
// delivers the key without a handshake).
func signalReady(descriptor int) error {
	if descriptor < 0 {
		return nil
	}
	if descriptor <= 2 {
		return errors.New("--ready-fd must be greater than 2")
	}
	file := os.NewFile(uintptr(descriptor), "apply-signer-ready")
	if file == nil {
		return errors.New("ready descriptor is invalid")
	}
	defer file.Close()
	if _, err := file.Write([]byte{1}); err != nil {
		return fmt.Errorf("could not signal readiness: %w", err)
	}
	return nil
}

// attachSignerSocket wraps the inherited descriptor and requires a connected
// local Unix stream socket, matching the supervisor's own descriptor contract.
func attachSignerSocket(descriptor int) (net.Conn, error) {
	if descriptor <= 2 {
		return nil, errors.New("--socket-fd must be greater than 2")
	}
	var metadata syscall.Stat_t
	if err := syscall.Fstat(descriptor, &metadata); err != nil {
		return nil, fmt.Errorf("could not inspect signer socket descriptor: %w", err)
	}
	if metadata.Mode&syscall.S_IFMT != syscall.S_IFSOCK {
		return nil, errors.New("--socket-fd must be a socket")
	}
	socketType, err := syscall.GetsockoptInt(descriptor, syscall.SOL_SOCKET, syscall.SO_TYPE)
	if err != nil {
		return nil, err
	}
	if socketType != syscall.SOCK_STREAM {
		return nil, errors.New("--socket-fd must be a stream socket")
	}
	if peer, err := syscall.Getpeername(descriptor); err != nil {
		return nil, fmt.Errorf("--socket-fd must be connected: %w", err)
	} else if _, ok := peer.(*syscall.SockaddrUnix); !ok {
		return nil, errors.New("--socket-fd must be a local Unix socket")
	}
	syscall.CloseOnExec(descriptor)
	file := os.NewFile(uintptr(descriptor), "apply-signer-socket")
	if file == nil {
		return nil, errors.New("signer socket descriptor is invalid")
	}
	connection, err := net.FileConn(file)
	_ = file.Close()
	if err != nil {
		return nil, fmt.Errorf("could not attach signer socket: %w", err)
	}
	if _, ok := connection.(*net.UnixConn); !ok {
		_ = connection.Close()
		return nil, errors.New("signer socket must be a Unix socket")
	}
	return connection, nil
}

// serveConnection runs the protocol loop until EOF or an unrecoverable frame
// error. It is deliberately hardening-free so tests can drive it in-process.
func (server *signerServer) serveConnection(connection net.Conn) error {
	for {
		var request signerRequest
		if err := receiveFrame(connection, &request); err != nil {
			zeroRequest(&request)
			if errors.Is(err, io.EOF) || errors.Is(err, io.ErrUnexpectedEOF) {
				return nil
			}
			// A frame we cannot even parse to an ID cannot be answered
			// coherently; drop the connection so the broker fails closed.
			return err
		}
		responded, err := server.handle(connection, &request)
		zeroRequest(&request)
		if err != nil {
			return err
		}
		if !responded {
			return errors.New("external signer produced no response")
		}
	}
}

// handle validates and answers exactly one request. It returns whether a
// response was written and any transport error. Validation failures are answered
// with an error frame (so the broker fails closed) and are not transport errors.
func (server *signerServer) handle(connection net.Conn, request *signerRequest) (bool, error) {
	if request.Version != signerProtocolVersion ||
		!request.hasField("version") || !request.hasField("id") || !request.hasField("op") {
		return server.reject(connection, request.ID, "external signer request is malformed")
	}
	if request.ID <= 0 || request.ID <= server.lastID {
		return server.reject(connection, request.ID,
			"external signer request ID must be positive and strictly increasing")
	}
	server.lastID = request.ID

	if request.Scope != server.scope {
		// The provisioned scope is the whole capability; a request for any other
		// scope (including a valid-but-different one) is refused.
		return server.reject(connection, request.ID,
			fmt.Sprintf("external signer only serves scope %q", server.scope))
	}

	switch request.Operation {
	case "challenge":
		if !request.hasExactFields("version", "id", "op", "scope", "challenge") {
			return server.reject(connection, request.ID, "external signer challenge is malformed")
		}
		return server.answerChallenge(connection, request)
	case "sign":
		if !request.hasExactFields("version", "id", "op", "scope", "payload") {
			return server.reject(connection, request.ID, "external signer sign request is malformed")
		}
		return server.answerSign(connection, request)
	default:
		return server.reject(connection, request.ID, "external signer operation is not supported")
	}
}

func (server *signerServer) answerChallenge(connection net.Conn, request *signerRequest) (bool, error) {
	message := make([]byte, 0, len(signerChallengeDomain)+len(server.scope)+1+len(request.Challenge))
	message = append(message, signerChallengeDomain...)
	message = append(message, server.scope...)
	message = append(message, 0)
	message = append(message, request.Challenge...)
	signature := ed25519.Sign(server.privateKey, message)
	zero(message)
	server.audit.challenge(server.scope, request.ID, request.Challenge)
	return true, sendFrame(connection, challengeResponse{
		Version:   signerProtocolVersion,
		ID:        request.ID,
		OK:        true,
		PublicKey: server.publicKey,
		Signature: signature,
	})
}

func (server *signerServer) answerSign(connection net.Conn, request *signerRequest) (bool, error) {
	message := make([]byte, 0, len(signerSignatureDomain)+len(server.scope)+1+len(request.Payload))
	message = append(message, signerSignatureDomain...)
	message = append(message, server.scope...)
	message = append(message, 0)
	message = append(message, request.Payload...)
	signature := ed25519.Sign(server.privateKey, message)
	messageDigest := sha256.Sum256(message)
	payloadDigest := sha256.Sum256(request.Payload)
	zero(message)
	server.signed++
	server.audit.sign(
		server.scope, request.ID, len(request.Payload),
		hex.EncodeToString(payloadDigest[:]),
		hex.EncodeToString(messageDigest[:]),
		bestEffortCitation(request.Payload),
	)
	return true, sendFrame(connection, signResponse{
		Version:   signerProtocolVersion,
		ID:        request.ID,
		OK:        true,
		Signature: signature,
	})
}

func (server *signerServer) reject(connection net.Conn, requestID int64, message string) (bool, error) {
	server.audit.refuse(server.scope, requestID, message)
	return true, sendFrame(connection, errorResponse{
		Version: signerProtocolVersion,
		ID:      requestID,
		OK:      false,
		Error:   message,
	})
}

func zeroRequest(request *signerRequest) {
	zero(request.Challenge)
	zero(request.Payload)
}

func supervisorPublicKind(info contextInfo) string {
	if info.local {
		return "local-dev"
	}
	return "ci"
}

// bestEffortCitation extracts a human-readable citation from a manifest payload
// for the audit line only. It never influences the sign decision (the payload is
// opaque to the signer by design) and is aggressively sanitized against log
// injection. An unparseable or citation-less payload yields "".
func bestEffortCitation(payload []byte) string {
	if len(payload) == 0 || len(payload) > maxFrameBytes {
		return ""
	}
	var probe struct {
		Citation string `json:"citation"`
	}
	// Ignore every decode error: a non-JSON or differently-shaped payload simply
	// has no citation to surface.
	if err := json.Unmarshal(payload, &probe); err != nil {
		return ""
	}
	return sanitizeAuditValue(probe.Citation)
}

const maxAuditCitationBytes = 120

func sanitizeAuditValue(value string) string {
	if value == "" {
		return ""
	}
	out := make([]byte, 0, len(value))
	for index := 0; index < len(value) && len(out) < maxAuditCitationBytes; index++ {
		character := value[index]
		switch {
		case character >= 'a' && character <= 'z',
			character >= 'A' && character <= 'Z',
			character >= '0' && character <= '9':
			out = append(out, character)
		case character == ' ' || character == '-' || character == '_' ||
			character == '.' || character == '/' || character == ':' ||
			character == '(' || character == ')' || character == ',':
			out = append(out, character)
		default:
			out = append(out, '.')
		}
	}
	return string(out)
}

// auditLog writes single-line, machine-parseable audit records. It never emits
// key material: only content hashes, scopes, request IDs, sanitized citations,
// and timestamps.
type auditLog struct {
	writer io.Writer
}

func (log *auditLog) line(fields string) {
	if log == nil || log.writer == nil {
		return
	}
	fmt.Fprintf(log.writer, "axiom-apply-signer %s ts=%s\n", fields, time.Now().UTC().Format(time.RFC3339))
}

func (log *auditLog) bind(scope, kind string, info contextInfo) {
	if info.local {
		log.line(fmt.Sprintf("event=bind scope=%s mode=%s", scope, kind))
		return
	}
	log.line(fmt.Sprintf(
		"event=bind scope=%s mode=%s repository=%s workflow_ref=%s event=%s sha=%s run_id=%s",
		scope, kind, info.repository, info.workflowRef, info.eventName, info.sha, info.runID,
	))
}

func (log *auditLog) challenge(scope string, requestID int64, nonce []byte) {
	digest := sha256.Sum256(nonce)
	log.line(fmt.Sprintf(
		"event=challenge scope=%s request_id=%d nonce_sha256=%s",
		scope, requestID, hex.EncodeToString(digest[:]),
	))
}

func (log *auditLog) sign(
	scope string, requestID int64, payloadBytes int,
	payloadSHA256, messageSHA256, citation string,
) {
	fields := fmt.Sprintf(
		"event=sign scope=%s request_id=%d payload_bytes=%d payload_sha256=%s signed_message_sha256=%s",
		scope, requestID, payloadBytes, payloadSHA256, messageSHA256,
	)
	if citation != "" {
		fields += " citation=" + citation
	}
	log.line(fields)
}

func (log *auditLog) refuse(scope string, requestID int64, reason string) {
	log.line(fmt.Sprintf(
		"event=refuse scope=%s request_id=%d reason=%q", scope, requestID, sanitizeAuditValue(reason),
	))
}

func (log *auditLog) shutdown(scope string, signed int) {
	log.line(fmt.Sprintf("event=shutdown scope=%s signatures=%d", scope, signed))
}
