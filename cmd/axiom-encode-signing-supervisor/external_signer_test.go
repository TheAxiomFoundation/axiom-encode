//go:build darwin || linux

package main

import (
	"bytes"
	"crypto/ed25519"
	"fmt"
	"net"
	"os"
	"testing"
	"time"

	"golang.org/x/sys/unix"
)

func TestConnectExternalSignerVerifiesChallengeSignature(t *testing.T) {
	for _, validSignature := range []bool{true, false} {
		t.Run(fmt.Sprintf("valid_signature=%t", validSignature), func(t *testing.T) {
			privateKey := ed25519.NewKeyFromSeed(bytes.Repeat([]byte{0xab}, ed25519.SeedSize))
			publicKey := privateKey.Public().(ed25519.PublicKey)
			descriptors, err := unix.Socketpair(unix.AF_UNIX, unix.SOCK_STREAM, 0)
			if err != nil {
				t.Fatal(err)
			}
			serverFile := os.NewFile(uintptr(descriptors[1]), "synthetic-external-signer")
			server, err := net.FileConn(serverFile)
			_ = serverFile.Close()
			if err != nil {
				_ = unix.Close(descriptors[0])
				t.Fatal(err)
			}
			defer server.Close()
			if err := server.SetDeadline(time.Now().Add(5 * time.Second)); err != nil {
				_ = unix.Close(descriptors[0])
				t.Fatal(err)
			}
			served := make(chan error, 1)
			go func() {
				var request signerRequest
				if err := receiveFrame(server, &request); err != nil {
					served <- err
					return
				}
				if request.Version != 2 || request.ID != 1 || request.Operation != "challenge" ||
					request.Scope != "apply_ed25519" || len(request.Challenge) != signerChallengeBytes {
					served <- fmt.Errorf("unexpected challenge request")
					return
				}
				message := append([]byte("axiom-encode/external-signer-challenge/v2\x00apply_ed25519\x00"), request.Challenge...)
				if !validSignature {
					message = []byte("wrong")
				}
				served <- sendFrame(server, signerResponse{
					Version:   2,
					ID:        request.ID,
					OK:        true,
					PublicKey: publicKey,
					Signature: ed25519.Sign(privateKey, message),
				})
			}()

			// This exercises the verifier directly, without racing a broker's
			// detailed stderr write against its supervising parent's cleanup.
			signer, connectErr := connectExternalSigner("apply", descriptors[0], publicKey)
			if signer != nil {
				defer signer.close()
			}
			if err := <-served; err != nil {
				t.Fatal(err)
			}
			if validSignature {
				if connectErr != nil || signer == nil {
					t.Fatalf("valid challenge rejected: %v", connectErr)
				}
			} else {
				if signer != nil || connectErr == nil || connectErr.Error() != "external apply signer challenge response is invalid" {
					t.Fatalf("wrong-message challenge must be rejected without a signer, got signer=%v, error=%v", signer != nil, connectErr)
				}
			}
		})
	}
}
