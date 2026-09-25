//go:build darwin || linux

package main

import (
	"bytes"
	"compress/zlib"
	"crypto/sha256"
	"encoding/base64"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"errors"
	"io"
	"os"
	"path/filepath"
	"strings"
	"syscall"
	"testing"
	"time"
)

func doctorTestRoot(t *testing.T) string {
	t.Helper()
	root, err := filepath.EvalSymlinks(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	return root
}
func doctorWrite(t *testing.T, root, name string, raw []byte) string {
	t.Helper()
	path := filepath.Join(root, name)
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, raw, 0o644); err != nil {
		t.Fatal(err)
	}
	return path
}
func doctorStatus(t *testing.T, err error, want string) {
	t.Helper()
	var problem *doctorProblem
	if !errors.As(err, &problem) || problem.status != want {
		t.Fatalf("want %s, got %v", want, err)
	}
}
func doctorCheckByID(t *testing.T, r doctorReport, id string) doctorCheck {
	t.Helper()
	for _, check := range r.Checks {
		if check.ID == id {
			return check
		}
	}
	t.Fatalf("check %s missing", id)
	return doctorCheck{}
}

func TestDoctorReadOnlyRegularInputs(t *testing.T) {
	root := doctorTestRoot(t)
	regular := doctorWrite(t, root, "regular.json", []byte(`{"public":"fixture"}`))
	raw, err := newDoctorBudget().file(regular, 128, false, false)
	if err != nil || string(raw) != `{"public":"fixture"}` {
		t.Fatalf("regular: %s %v", raw, err)
	}
	_, err = newDoctorBudget().file(filepath.Join(root, "absent"), 128, false, false)
	doctorStatus(t, err, "missing")
	_, err = newDoctorBudget().file(root, 128, false, false)
	doctorStatus(t, err, "untrusted")
	symlink := filepath.Join(root, "symlink")
	if err := os.Symlink(regular, symlink); err != nil {
		t.Fatal(err)
	}
	_, err = newDoctorBudget().file(symlink, 128, false, false)
	doctorStatus(t, err, "untrusted")
	fifo := filepath.Join(root, "fifo")
	if err := syscall.Mkfifo(fifo, 0o600); err != nil {
		t.Fatal(err)
	}
	_, err = newDoctorBudget().file(fifo, 128, false, false)
	doctorStatus(t, err, "untrusted")
	_, err = newDoctorBudget().file(regular, 128, true, false)
	doctorStatus(t, err, "malformed")
	after, _ := os.ReadFile(regular)
	if !bytes.Equal(raw, after) {
		t.Fatal("doctor modified its input")
	}
}

func TestDoctorOpenRejectsChangedAncestor(t *testing.T) {
	root := doctorTestRoot(t)
	public := doctorWrite(t, root, "public/data", []byte("public"))
	secret := doctorWrite(t, root, "private/data", []byte("CREDENTIAL_MUST_NOT_BE_READ"))
	if err := doctorPath(public, false, false); err != nil {
		t.Fatal(err)
	}
	if err := os.Rename(filepath.Dir(public), filepath.Join(root, "old-public")); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(filepath.Dir(secret), filepath.Dir(public)); err != nil {
		t.Fatal(err)
	}
	f, err := doctorOpen(public, false)
	if err == nil {
		f.Close()
		t.Fatal("opened a replaced ancestor symlink")
	}
}

func TestDoctorBudgets(t *testing.T) {
	root := doctorTestRoot(t)
	path := doctorWrite(t, root, "large", []byte("12345"))
	_, err := newDoctorBudget().file(path, 4, false, false)
	doctorStatus(t, err, "incomplete")
	for _, budget := range []*doctorBudget{
		{100, 4, time.Now().Add(time.Second)},
		{0, 100, time.Now().Add(time.Second)},
		{100, 100, time.Now().Add(-time.Second)},
	} {
		_, err := budget.file(path, 10, false, false)
		doctorStatus(t, err, "incomplete")
	}
}

func TestDoctorPublicJSONDoesNotEchoValues(t *testing.T) {
	for _, raw := range []string{
		`{"secret":"CREDENTIAL_MARKER"}`,
		`{"schema":"a","schema":"CREDENTIAL_MARKER"}`,
		`{"schema":null} {"secret":"CREDENTIAL_MARKER"}`,
		`{"schema":CREDENTIAL_MARKER}`,
	} {
		var value struct {
			Schema string `json:"schema"`
		}
		err := doctorJSON([]byte(raw), &value)
		doctorStatus(t, err, "malformed")
		if strings.Contains(err.Error(), "CREDENTIAL_MARKER") {
			t.Fatal("parser leaked input")
		}
	}
}

func TestDoctorArgumentsAndHelpAreNonexecuting(t *testing.T) {
	for _, args := range [][]string{
		{"--apply-signer-fd", "CREDENTIAL_MARKER"},
		{"--codex-subscription-auth", "CREDENTIAL_MARKER"},
		{"--json=CREDENTIAL_MARKER"},
		{"--installation", "relative/CREDENTIAL_MARKER"},
		{"--corpus-root", "/corpus", "--release-name", "../../CREDENTIAL_MARKER"},
		{"--expected-encoder-commit", "CREDENTIAL_MARKER"},
		{"--expected-supervisor-sha256", "CREDENTIAL_MARKER"},
		{"--", "encode", "CREDENTIAL_MARKER"},
	} {
		var stdout, stderr bytes.Buffer
		if rc := runDoctor(args, &stdout, &stderr); rc != 64 {
			t.Fatalf("invalid args returned %d", rc)
		}
		if strings.Contains(stdout.String()+stderr.String(), "CREDENTIAL_MARKER") {
			t.Fatal("flag parser leaked value")
		}
	}
	var stdout, stderr bytes.Buffer
	if rc := runDoctor([]string{"--help"}, &stdout, &stderr); rc != 0 {
		t.Fatalf("help: %d", rc)
	}
	if !strings.Contains(stdout.String(), "never admission") {
		t.Fatal("missing scope")
	}
}

func TestDoctorKnownBlockersAreStructuredAndRedacted(t *testing.T) {
	t.Setenv("AXIOM_ENCODE_APPLY_SIGNING_PRIVATE_KEY", "CREDENTIAL_MARKER")
	t.Setenv("AXIOM_ENCODE_SIGNING_BROKER_FD", "CREDENTIAL_MARKER")
	var stdout, stderr bytes.Buffer
	rc := runDoctor([]string{"--installation", "/axiom-doctor-known-missing-installation", "--json"}, &stdout, &stderr)
	if rc != 1 {
		t.Fatalf("want blocked exit 1, got %d", rc)
	}
	if strings.Contains(stdout.String()+stderr.String(), "CREDENTIAL_MARKER") {
		t.Fatal("environment leaked")
	}
	var report doctorReport
	if err := json.Unmarshal(stdout.Bytes(), &report); err != nil {
		t.Fatal(err)
	}
	if report.Schema != doctorSchema || report.Status != "blocked" || report.Admission != "not_checked" {
		t.Fatalf("wrong report: %+v", report)
	}
	if c := doctorCheckByID(t, report, "supervisor"); c.Status != "missing" {
		t.Fatalf("supervisor: %+v", c)
	}
	if c := doctorCheckByID(t, report, "environment"); c.Status != "malformed" {
		t.Fatalf("environment: %+v", c)
	}
	for _, id := range []string{"subscription_credentials", "apply_signer", "release_admission", "runtime_admission"} {
		if c := doctorCheckByID(t, report, id); c.Status != "not_checked" {
			t.Fatalf("%s: %+v", id, c)
		}
	}
}

func TestDoctorReleasePresenceNeverConfersTrust(t *testing.T) {
	root := doctorTestRoot(t)
	name, digest := "fixture-release", strings.Repeat("a", 64)
	path := doctorWrite(t, root, "releases/"+name+"/"+digest+".json", []byte(`{"schema_version":"axiom-corpus/release-object/v3","release":"fixture-release","content_sha256":"`+digest+`","content":{"not":"validated"},"signature":{"value":"NOT_A_REAL_SIGNATURE"}}`))
	o := doctorOptions{installation: "/axiom-doctor-known-missing-installation", corpusRoot: root, releaseName: name, releaseDigest: digest}
	report := inspectDoctor(o, newDoctorBudget())
	if c := doctorCheckByID(t, report, "corpus_release"); c.Status != "observed" || !strings.Contains(c.Detail, "NOT verified") {
		t.Fatalf("presence overclaim: %+v", c)
	}
	if report.Admission != "not_checked" {
		t.Fatal("file presence conferred admission")
	}
	if err := os.WriteFile(path, []byte(`{"private":"CREDENTIAL_MARKER"}`), 0o644); err != nil {
		t.Fatal(err)
	}
	report = inspectDoctor(o, newDoctorBudget())
	if c := doctorCheckByID(t, report, "corpus_release"); c.Status != "malformed" {
		t.Fatalf("malformed: %+v", c)
	}
	raw, _ := json.Marshal(report)
	if bytes.Contains(raw, []byte("CREDENTIAL_MARKER")) {
		t.Fatal("release leaked content")
	}
}

func TestDoctorProductionRejectsUserOwnedConfiguration(t *testing.T) {
	if trustPolicyAllowsWritablePath() || os.Geteuid() == 0 {
		t.Skip("production unprivileged policy only")
	}
	root := doctorTestRoot(t)
	path := doctorWrite(t, root, "signing-trust-roots.json", []byte(`{"secret":"CREDENTIAL_MARKER"}`))
	_, err := newDoctorBudget().file(path, 65536, false, true)
	doctorStatus(t, err, "untrusted")
}

// Static distinct public bytes only: no key generation or signatures.
func doctorPublicRoots(t *testing.T) []byte {
	t.Helper()
	key := func(value byte) string { return base64.StdEncoding.EncodeToString(bytes.Repeat([]byte{value}, 32)) }
	raw, err := json.Marshal(map[string]any{"schema": "axiom-encode/signing-trust-roots/v3", "apply_ed25519_public_key": key(1), "eval_ed25519_public_key": key(2), "corpus_release_ed25519_public_keys": []string{key(3), key(4)}})
	if err != nil {
		t.Fatal(err)
	}
	return raw
}

func TestDoctorProtectedFixtures(t *testing.T) {
	if !trustPolicyAllowsWritablePath() {
		t.Skip("fixture build only; never install")
	}
	root := doctorTestRoot(t)
	path := doctorWrite(t, root, "signing-trust-roots.json", doctorPublicRoots(t))
	o := doctorOptions{installation: root}
	report := inspectDoctor(o, newDoctorBudget())
	if c := doctorCheckByID(t, report, "public_trust_roots"); c.Status != "observed" {
		t.Fatalf("present: %+v", c)
	}
	if report.Admission != "not_checked" || report.BuildKind != "test-fixture-nonpublishable" {
		t.Fatal("fixture authority overclaim")
	}
	for _, raw := range [][]byte{
		[]byte(`{"schema":"unknown","value":"CREDENTIAL_MARKER"}`),
		bytes.ReplaceAll(doctorPublicRoots(t), []byte(base64.StdEncoding.EncodeToString(bytes.Repeat([]byte{2}, 32))), []byte(base64.StdEncoding.EncodeToString(bytes.Repeat([]byte{1}, 32)))),
	} {
		if err := os.WriteFile(path, raw, 0o644); err != nil {
			t.Fatal(err)
		}
		report = inspectDoctor(o, newDoctorBudget())
		if c := doctorCheckByID(t, report, "public_trust_roots"); c.Status != "malformed" {
			t.Fatalf("malformed: %+v", c)
		}
	}
	if err := os.Chmod(path, 0o666); err != nil {
		t.Fatal(err)
	}
	report = inspectDoctor(o, newDoctorBudget())
	if c := doctorCheckByID(t, report, "public_trust_roots"); c.Status != "untrusted" {
		t.Fatalf("writable: %+v", c)
	}
}

func TestDoctorPackageHashMatchesProvisionerAlgorithm(t *testing.T) {
	if !trustPolicyAllowsWritablePath() {
		t.Skip("fixture build only")
	}
	root := doctorTestRoot(t)
	doctorWrite(t, root, "z.py", []byte("z"))
	doctorWrite(t, root, "a/b.py", []byte("b"))
	doctorWrite(t, root, "__pycache__/ignored.pyc", []byte("ignored"))
	digest := sha256.New()
	digest.Write([]byte("axiom-eval-tree-v1\x00"))
	if err := newDoctorBudget().tree(root, digest, "", 0); err != nil {
		t.Fatal(err)
	}
	// Independently computed by scripts/provision_verification_supervisor.py's
	// _deterministic_package_tree_sha256; files precede sorted subdirectories.
	const want = "e23d1bb4fec5b35bdbb8f3b31f8d2d197d55e7e871d125c830ccbd3567fe04f8"
	if hex.EncodeToString(digest.Sum(nil)) != want {
		t.Fatalf("package digest: %x", digest.Sum(nil))
	}
	doctorWrite(t, root, "inject.pth", []byte("import os"))
	doctorStatus(t, newDoctorBudget().tree(root, nil, "", 0), "untrusted")
}

func doctorRuntimeFixture(t *testing.T) (string, doctorAttestation) {
	t.Helper()
	root := doctorTestRoot(t)
	runtimeRoot := filepath.Join(root, "python")
	interpreter := filepath.Join(runtimeRoot, "bin", "python3.13")
	native := []byte{0xcf, 0xfa, 0xed, 0xfe} // header only; NEVER executed
	doctorWrite(t, root, "python/bin/python3.13", native)
	doctorWrite(t, root, "axiom-encode", []byte("#!"+interpreter+" -I\nraise SystemExit('launcher executed')\n"))
	doctorWrite(t, root, "python/bin/git", []byte("#!"+interpreter+" -I\nraise SystemExit('git executed')\n"))
	for _, path := range []string{interpreter, filepath.Join(root, "axiom-encode"), filepath.Join(runtimeRoot, "bin", "git")} {
		if err := os.Chmod(path, 0o755); err != nil {
			t.Fatal(err)
		}
	}
	packageRoot := filepath.Join(runtimeRoot, "lib", "python3.13", "site-packages", "axiom_encode")
	doctorWrite(t, packageRoot, "__init__.py", []byte("__version__ = '0.2.1'\n"))
	doctorWrite(t, packageRoot, "_trusted_signing_bootstrap.py", []byte("raise SystemExit('bootstrap executed')\n"))
	h := sha256.New()
	h.Write([]byte("axiom-eval-tree-v1\x00"))
	if err := newDoctorBudget().tree(packageRoot, h, "", 0); err != nil {
		t.Fatal(err)
	}
	a := doctorAttestation{Schema: "axiom-encode/trusted-runtime-attestation/v1", ProvisionedAt: "2026-09-07T00:00:00Z"}
	a.Encoder.Origin = "github.com/TheAxiomFoundation/axiom-encode"
	a.Encoder.Commit = strings.Repeat("a", 40)
	a.Encoder.Version = "0.2.1"
	a.Encoder.TreeSHA256 = hex.EncodeToString(h.Sum(nil))
	codex := doctorWrite(t, root, "bin/codex", native)
	if err := os.Chmod(codex, 0o755); err != nil {
		t.Fatal(err)
	}
	sum := sha256.Sum256(native)
	config := trustedCodexCLI{Schema: "axiom-encode/trusted-codex-cli/v1", Path: codex, Version: "0.144.0", SHA256: hex.EncodeToString(sum[:])}
	raw, _ := json.Marshal(config)
	doctorWrite(t, root, "codex-cli.json", raw)
	// Marshal directly to avoid giving test helpers any private-key type.
	attestationRaw, _ := json.Marshal(a)
	var payload map[string]any
	if err := json.Unmarshal(attestationRaw, &payload); err != nil {
		t.Fatal(err)
	}
	payload["codex_cli"] = map[string]string{"version": config.Version, "sha256": config.SHA256}
	attestationRaw, _ = json.Marshal(payload)
	if err := json.Unmarshal(attestationRaw, &a); err != nil {
		t.Fatal(err)
	}
	doctorWrite(t, runtimeRoot, "runtime-attestation.json", attestationRaw)
	return root, a
}

func TestDoctorRuntimeAndCodexIdentityFixtures(t *testing.T) {
	if !trustPolicyAllowsWritablePath() {
		t.Skip("fixture build only")
	}
	root, a := doctorRuntimeFixture(t)
	options := doctorOptions{installation: root, expectedCommit: a.Encoder.Commit}
	r := inspectDoctor(options, newDoctorBudget())
	for _, id := range []string{"python_launcher", "python_runtime_tree", "runtime_git_wrapper", "runtime_attestation", "encoder_package", "subscription_codex"} {
		if c := doctorCheckByID(t, r, id); c.Status != "observed" {
			t.Fatalf("%s: %+v", id, c)
		}
	}
	if r.Admission != "not_checked" {
		t.Fatal("fixture admitted")
	}
	options.expectedCommit = strings.Repeat("b", 40)
	r = inspectDoctor(options, newDoctorBudget())
	if c := doctorCheckByID(t, r, "runtime_attestation"); c.Status != "stale" {
		t.Fatalf("stale: %+v", c)
	}
	doctorWrite(t, root, "python/lib/python3.13/site-packages/axiom_encode/__init__.py", []byte("modified package"))
	r = inspectDoctor(options, newDoctorBudget())
	if c := doctorCheckByID(t, r, "encoder_package"); c.Status != "untrusted" {
		t.Fatalf("package digest: %+v", c)
	}
	doctorWrite(t, root, "bin/codex", []byte{0xcf, 0xfa, 0xed, 0xfe, 1})
	r = inspectDoctor(options, newDoctorBudget())
	if c := doctorCheckByID(t, r, "subscription_codex"); c.Status != "untrusted" {
		t.Fatalf("codex digest: %+v", c)
	}
	doctorWrite(t, root, "python/runtime-attestation.json", []byte(`{"schema":"CREDENTIAL_MARKER"}`))
	r = inspectDoctor(options, newDoctorBudget())
	if c := doctorCheckByID(t, r, "runtime_attestation"); c.Status != "malformed" {
		t.Fatalf("attestation: %+v", c)
	}
}

func TestDoctorDirectoryBudgetAndSpecialEntries(t *testing.T) {
	if !trustPolicyAllowsWritablePath() {
		t.Skip("fixture build only")
	}
	root := doctorTestRoot(t)
	for _, name := range []string{"a", "b", "c"} {
		doctorWrite(t, root, name, []byte("fixture"))
	}
	budget := newDoctorBudget()
	budget.entries = 2
	doctorStatus(t, budget.tree(root, nil, "", 0), "incomplete")
	if err := os.Symlink(root, filepath.Join(root, "cycle")); err != nil {
		t.Fatal(err)
	}
	doctorStatus(t, newDoctorBudget().tree(root, nil, "", 0), "untrusted")
}

func TestDoctorRejectsCaseVariantPublicJSONFields(t *testing.T) {
	for _, raw := range []string{
		`{"SCHEMA":"x"}`,
		`{"schema":"x","SCHEMA":"y"}`,
		`{"axiom_encode":{"COMMIT":"a"}}`,
		`{"axiom_encode":{"commit":"a","COMMIT":"b"}}`,
		`{"codex_cli":{"SHA256":"CREDENTIAL_MARKER"}}`,
	} {
		var a doctorAttestation
		err := doctorJSON([]byte(raw), &a)
		doctorStatus(t, err, "malformed")
		if strings.Contains(err.Error(), "CREDENTIAL_MARKER") {
			t.Fatal("input leaked")
		}
	}
	if !trustPolicyAllowsWritablePath() {
		return
	}
	root, _ := doctorRuntimeFixture(t)
	path := filepath.Join(root, "python/runtime-attestation.json")
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	for _, altered := range [][]byte{
		bytes.Replace(raw, []byte(`"commit"`), []byte(`"COMMIT"`), 1),
		bytes.Replace(raw, []byte(`"commit":`), []byte(`"COMMIT":"`+strings.Repeat("b", 40)+`","commit":`), 1),
	} {
		if err := os.WriteFile(path, altered, 0o644); err != nil {
			t.Fatal(err)
		}
		r := inspectDoctor(doctorOptions{installation: root}, newDoctorBudget())
		if c := doctorCheckByID(t, r, "runtime_attestation"); c.Status != "malformed" {
			t.Fatalf("case alias accepted: %+v", c)
		}
		if c := doctorCheckByID(t, r, "encoder_package"); c.Status != "incomplete" {
			t.Fatalf("invalid identity used: %+v", c)
		}
	}
}

type doctorFailingWriter struct{}

func (doctorFailingWriter) Write([]byte) (int, error) { return 0, io.ErrClosedPipe }
func TestDoctorOutputErrors(t *testing.T) {
	for _, args := range [][]string{{"--help"}, {"--installation", "/axiom-doctor-known-missing-installation"}, {"--installation", "/axiom-doctor-known-missing-installation", "--json"}} {
		if rc := runDoctor(args, doctorFailingWriter{}, io.Discard); rc != 64 {
			t.Fatalf("output failure returned %d", rc)
		}
	}
}

func TestDoctorCompressedELFMetadataIsNeverParsed(t *testing.T) {
	if !trustPolicyAllowsWritablePath() {
		t.Skip("fixture build only")
	}
	// Tiny ELF64 with a section-name table declaring a 1 GiB uncompressed size.
	// A generic debug/elf reader can allocate/decompress outside our file budget.
	// The doctor must only inspect the native magic and hash these bounded bytes.
	var compressed bytes.Buffer
	zw := zlib.NewWriter(&compressed)
	if _, err := zw.Write([]byte("\x00.shstrtab\x00")); err != nil {
		t.Fatal(err)
	}
	if err := zw.Close(); err != nil {
		t.Fatal(err)
	}
	raw := make([]byte, 64+2*64+24+compressed.Len())
	copy(raw, []byte{0x7f, 'E', 'L', 'F', 2, 1, 1})
	put16 := func(offset int, value uint16) { binary.LittleEndian.PutUint16(raw[offset:], value) }
	put32 := func(offset int, value uint32) { binary.LittleEndian.PutUint32(raw[offset:], value) }
	put64 := func(offset int, value uint64) { binary.LittleEndian.PutUint64(raw[offset:], value) }
	put16(16, 2)
	put16(18, 62)
	put32(20, 1)
	put64(40, 64)
	put16(52, 64)
	put16(58, 64)
	put16(60, 2)
	put16(62, 1)
	sh := 128
	put32(sh+4, 3)
	put64(sh+8, 0x800)
	put64(sh+24, 192)
	put64(sh+32, uint64(24+compressed.Len()))
	put64(sh+48, 1)
	put32(192, 1)
	put64(200, 1<<30)
	put64(208, 1)
	copy(raw[216:], compressed.Bytes())
	root := doctorTestRoot(t)
	path := doctorWrite(t, root, "axiom-encode-signing-supervisor", raw)
	if err := os.Chmod(path, 0o755); err != nil {
		t.Fatal(err)
	}
	r := inspectDoctor(doctorOptions{installation: root}, newDoctorBudget())
	c := doctorCheckByID(t, r, "supervisor")
	if c.Status != "observed" || !strings.Contains(c.Detail, "not executed or parsed") {
		t.Fatalf("metadata parsed/overclaimed: %+v", c)
	}
	if c := doctorCheckByID(t, r, "supervisor_provenance"); c.Status != "not_checked" {
		t.Fatalf("header conferred provenance: %+v", c)
	}
	digest := sha256.Sum256(raw)
	r = inspectDoctor(doctorOptions{installation: root, expectedSupervisor: hex.EncodeToString(digest[:])}, newDoctorBudget())
	if c := doctorCheckByID(t, r, "supervisor"); c.Status != "observed" {
		t.Fatalf("bounded hash: %+v", c)
	}
	r = inspectDoctor(doctorOptions{installation: root, expectedSupervisor: strings.Repeat("b", 64)}, newDoctorBudget())
	if c := doctorCheckByID(t, r, "supervisor"); c.Status != "stale" {
		t.Fatalf("hash mismatch: %+v", c)
	}
}

func TestDoctorRecordsSelectionAndSubscriptionEnvironment(t *testing.T) {
	t.Setenv("CODEX_HOME", "CREDENTIAL_MARKER")
	o := doctorOptions{installation: "/axiom-doctor-known-missing-installation", expectedCommit: strings.Repeat("c", 40), corpusRoot: "/corpus", releaseName: "fixture-release", releaseDigest: strings.Repeat("d", 64)}
	r := inspectDoctor(o, newDoctorBudget())
	if r.ExpectedEncoderCommit != o.expectedCommit || r.CorpusSelection == nil || r.CorpusSelection.Name != o.releaseName || r.CorpusSelection.ContentSHA256 != o.releaseDigest || r.CorpusSelection.ObjectPath != "/corpus/releases/fixture-release/"+o.releaseDigest+".json" {
		t.Fatal("report lost selection")
	}
	if c := doctorCheckByID(t, r, "subscription_environment"); c.Status != "malformed" {
		t.Fatalf("missed launch blocker: %+v", c)
	}
	raw, _ := json.Marshal(r)
	if bytes.Contains(raw, []byte("CREDENTIAL_MARKER")) {
		t.Fatal("environment value leaked")
	}
	t.Setenv("CODEX_HOME", "")
	r = inspectDoctor(o, newDoctorBudget())
	if c := doctorCheckByID(t, r, "subscription_environment"); c.Status != "observed" {
		t.Fatalf("empty env falsely blocked: %+v", c)
	}
}
