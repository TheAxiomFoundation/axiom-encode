//go:build darwin || linux

package main

import (
	"bytes"
	"crypto/ed25519"
	"crypto/rand"
	"crypto/x509"
	"encoding/base64"
	"encoding/pem"
	"os"
	"path/filepath"
	"strings"
	"syscall"
	"testing"
)

func newSeedBase64(t *testing.T) (string, ed25519.PublicKey) {
	t.Helper()
	public, private, err := ed25519.GenerateKey(rand.Reader)
	if err != nil {
		t.Fatal(err)
	}
	return base64.StdEncoding.EncodeToString(private.Seed()), public
}

func TestParseRawSeedBase64(t *testing.T) {
	seedB64, public := newSeedBase64(t)
	key, err := parseEd25519PrivateKey([]byte(seedB64 + "\n"))
	if err != nil {
		t.Fatalf("parse raw seed: %v", err)
	}
	if !bytes.Equal(key.Public().(ed25519.PublicKey), public) {
		t.Fatal("parsed key derives the wrong public key")
	}
}

func TestParsePKCS8PEM(t *testing.T) {
	public, private, err := ed25519.GenerateKey(rand.Reader)
	if err != nil {
		t.Fatal(err)
	}
	der, err := x509.MarshalPKCS8PrivateKey(private)
	if err != nil {
		t.Fatal(err)
	}
	pemBytes := pem.EncodeToMemory(&pem.Block{Type: "PRIVATE KEY", Bytes: der})
	key, err := parseEd25519PrivateKey(pemBytes)
	if err != nil {
		t.Fatalf("parse PKCS8 PEM: %v", err)
	}
	if !bytes.Equal(key.Public().(ed25519.PublicKey), public) {
		t.Fatal("PKCS8 key derives the wrong public key")
	}
}

func TestRejectsLegacyHMACSecret(t *testing.T) {
	// A 48-byte random secret is exactly the stale HMAC-era key shape; it must
	// fail closed rather than be treated as an Ed25519 key.
	hmac := make([]byte, 48)
	if _, err := rand.Read(hmac); err != nil {
		t.Fatal(err)
	}
	_, err := parseEd25519PrivateKey([]byte(base64.StdEncoding.EncodeToString(hmac)))
	if err == nil || !strings.Contains(err.Error(), "seed bytes") {
		t.Fatalf("expected HMAC rejection, got %v", err)
	}
}

func TestRejectsGarbageKey(t *testing.T) {
	if _, err := parseEd25519PrivateKey([]byte("not a key at all !!!")); err == nil {
		t.Fatal("expected garbage key rejection")
	}
}

func TestEphemeralKeyIsValid(t *testing.T) {
	key, err := generateEphemeralKey()
	if err != nil {
		t.Fatal(err)
	}
	message := []byte("smoke")
	signature := ed25519.Sign(key, message)
	if !ed25519.Verify(key.Public().(ed25519.PublicKey), message, signature) {
		t.Fatal("ephemeral key did not round-trip a signature")
	}
}

func TestIngestFromPipeAccepted(t *testing.T) {
	seedB64, public := newSeedBase64(t)
	reader, writer, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	go func() {
		_, _ = writer.Write([]byte(seedB64))
		_ = writer.Close()
	}()
	duplicate, err := syscall.Dup(int(reader.Fd()))
	if err != nil {
		t.Fatal(err)
	}
	_ = reader.Close()
	key, err := ingestPrivateKeyFromDescriptor(duplicate)
	if err != nil {
		t.Fatalf("ingest from pipe: %v", err)
	}
	if !bytes.Equal(key.Public().(ed25519.PublicKey), public) {
		t.Fatal("ingested key derives the wrong public key")
	}
}

func TestIngestFromRegularFileRejected(t *testing.T) {
	seedB64, _ := newSeedBase64(t)
	path := filepath.Join(t.TempDir(), "key.b64")
	if err := os.WriteFile(path, []byte(seedB64), 0o600); err != nil {
		t.Fatal(err)
	}
	file, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer file.Close()
	_, err = ingestPrivateKeyFromDescriptor(int(file.Fd()))
	if err == nil || !strings.Contains(err.Error(), "pipe or socket") {
		t.Fatalf("expected regular-file rejection, got %v", err)
	}
}
