//go:build darwin || linux

package main

import (
	"bytes"
	"crypto/ed25519"
	"crypto/rand"
	"crypto/x509"
	"encoding/base64"
	"encoding/pem"
	"errors"
	"fmt"
	"io"
	"os"
	"syscall"
)

// maxKeyMaterialBytes bounds the private-key read. A raw base64 seed is 44
// bytes; a PKCS8 PEM is a few hundred. 16 KiB is a generous ceiling that still
// refuses an attacker who wired a firehose to the key descriptor.
const maxKeyMaterialBytes = 16 * 1024

// ingestPrivateKeyFromDescriptor reads Ed25519 private-key material from an
// already-open descriptor and returns the parsed key. The descriptor must be a
// pipe or socket, never a regular file: the key is streamed in by the launcher,
// never opened from a path this process controls and never re-readable from
// disk. The descriptor is drained and closed here; the raw material is zeroized
// before return.
func ingestPrivateKeyFromDescriptor(descriptor int) (ed25519.PrivateKey, error) {
	if descriptor <= 2 {
		return nil, errors.New("key descriptor must be greater than 2")
	}
	var metadata syscall.Stat_t
	if err := syscall.Fstat(descriptor, &metadata); err != nil {
		return nil, fmt.Errorf("could not inspect key descriptor: %w", err)
	}
	switch metadata.Mode & syscall.S_IFMT {
	case syscall.S_IFIFO, syscall.S_IFSOCK:
		// A streamed secret: acceptable.
	default:
		return nil, errors.New(
			"key descriptor must be a pipe or socket; a regular file, tty, or " +
				"device is refused so the key is never read from a persistent path",
		)
	}
	file := os.NewFile(uintptr(descriptor), "apply-signer-key")
	if file == nil {
		return nil, errors.New("key descriptor is invalid")
	}
	defer file.Close()
	raw, err := io.ReadAll(io.LimitReader(file, maxKeyMaterialBytes+1))
	if err != nil {
		zero(raw)
		return nil, fmt.Errorf("could not read private key material: %w", err)
	}
	defer zero(raw)
	if len(raw) > maxKeyMaterialBytes {
		return nil, errors.New("private key material exceeds its size limit")
	}
	return parseEd25519PrivateKey(raw)
}

// parseEd25519PrivateKey accepts two documented formats: a PKCS8 PEM block, or a
// base64 encoding of the 32-byte raw seed (the format used by the eval and
// corpus-release keys). Legacy HMAC secrets and any non-Ed25519 key are refused
// so a stale key store fails closed rather than producing an unverifiable
// signature.
func parseEd25519PrivateKey(material []byte) (ed25519.PrivateKey, error) {
	normalized := bytes.TrimSpace(material)
	if bytes.HasPrefix(normalized, []byte("-----BEGIN")) {
		block, rest := pem.Decode(normalized)
		if block == nil {
			return nil, errors.New("private key PEM is malformed")
		}
		defer zero(block.Bytes)
		if len(bytes.TrimSpace(rest)) != 0 {
			return nil, errors.New("private key PEM contains trailing data")
		}
		parsed, err := x509.ParsePKCS8PrivateKey(block.Bytes)
		if err != nil {
			return nil, errors.New("private key PEM must contain a PKCS8 key")
		}
		key, ok := parsed.(ed25519.PrivateKey)
		if !ok {
			return nil, errors.New("private key must be Ed25519")
		}
		return append(ed25519.PrivateKey(nil), key...), nil
	}
	decoded := make([]byte, base64.StdEncoding.DecodedLen(len(normalized)))
	count, err := base64.StdEncoding.Strict().Decode(decoded, normalized)
	if err != nil {
		zero(decoded)
		return nil, errors.New(
			"private key must be a PKCS8 PEM or the base64-encoded 32-byte Ed25519 seed",
		)
	}
	if count != ed25519.SeedSize {
		zero(decoded)
		return nil, fmt.Errorf(
			"base64 private key must decode to exactly %d seed bytes (got %d); "+
				"a legacy HMAC secret is not an Ed25519 key",
			ed25519.SeedSize, count,
		)
	}
	seed := decoded[:count]
	defer zero(seed)
	return ed25519.NewKeyFromSeed(seed), nil
}

// generateEphemeralKey mints a throwaway keypair for local development. Its
// public half is surfaced so a developer can build a matching trust root; its
// private half never leaves the process and is discarded on exit. This is the
// throwaway-key guard: local-dev mode never ingests external key material at
// all, so it is structurally impossible to sign with a production key outside
// GitHub Actions.
func generateEphemeralKey() (ed25519.PrivateKey, error) {
	_, private, err := ed25519.GenerateKey(rand.Reader)
	if err != nil {
		return nil, fmt.Errorf("could not generate ephemeral development key: %w", err)
	}
	return private, nil
}

func rawPublicKey(private ed25519.PrivateKey) []byte {
	public := private.Public().(ed25519.PublicKey)
	return append([]byte(nil), public...)
}
