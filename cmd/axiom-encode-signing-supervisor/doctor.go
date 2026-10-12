//go:build darwin || linux

package main

// This is an offline observation tool, never an admission or signing authority.
// In particular, it must not execute the installation it inspects. Public input
// failures use fixed messages: neither JSON values nor parser errors are logged.

import (
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"hash"
	"io"
	"os"
	"path/filepath"
	"reflect"
	"regexp"
	"sort"
	"strings"
	"time"

	"golang.org/x/sys/unix"
)

const doctorSchema = "axiom-encode/runtime-readiness/v1"
const doctorInstallOwner = "runtime operator (human install custody; docs/trusted-signing-supervisor.md)"
const doctorCorpusOwner = "axiom-corpus release owner (source review, signed release, local materialization)"

var doctorDigest = regexp.MustCompile(`^[0-9a-f]{64}$`)
var doctorCommit = regexp.MustCompile(`^[0-9a-f]{40}$`)
var doctorVersion = regexp.MustCompile(`^[0-9]+\.[0-9]+\.[0-9]+$`)
var doctorRelease = regexp.MustCompile(`^[a-z0-9]+(?:-[a-z0-9]+)*$`)

type doctorCheck struct {
	ID     string `json:"id"`
	Status string `json:"status"`
	Detail string `json:"detail"`
	Owner  string `json:"owner"`
	Action string `json:"action"`
}

type doctorReport struct {
	Schema                   string                 `json:"schema"`
	Status                   string                 `json:"status"`
	Admission                string                 `json:"admission"`
	BuildKind                string                 `json:"diagnostic_build_kind"`
	Installation             string                 `json:"installation"`
	ExpectedEncoderCommit    string                 `json:"expected_encoder_commit,omitempty"`
	ExpectedSupervisorSHA256 string                 `json:"expected_supervisor_sha256,omitempty"`
	CorpusSelection          *doctorCorpusSelection `json:"corpus_selection,omitempty"`
	Checks                   []doctorCheck          `json:"checks"`
	Limits                   map[string]int64       `json:"limits"`
}

type doctorCorpusSelection struct {
	Root          string `json:"root"`
	Name          string `json:"name"`
	ContentSHA256 string `json:"content_sha256"`
	ObjectPath    string `json:"object_path"`
}

type doctorOptions struct {
	installation       string
	expectedCommit     string
	expectedSupervisor string
	corpusRoot         string
	releaseName        string
	releaseDigest      string
	json               bool
}

type doctorProblem struct{ status, detail string }

func (p *doctorProblem) Error() string       { return p.detail }
func doctorFail(status, detail string) error { return &doctorProblem{status, detail} }

// Limits are shared across ALL checks, not renewed per file or subtree. Local
// filesystem syscalls themselves are not interruptible; this is a work budget,
// not a hard wall-clock guarantee on a stalled filesystem.
type doctorBudget struct {
	entries        int64
	remainingBytes int64
	deadline       time.Time
}

func newDoctorBudget() *doctorBudget {
	return &doctorBudget{50000, 512 * 1024 * 1024, time.Now().Add(10 * time.Second)}
}
func (b *doctorBudget) step() error {
	if time.Now().After(b.deadline) || b.entries <= 0 || b.remainingBytes < 0 {
		return doctorFail("incomplete", "Inspection budget exhausted; no readiness conclusion is available.")
	}
	b.entries--
	return nil
}
func (b *doctorBudget) read(file *os.File, limit int64) ([]byte, error) {
	if err := b.step(); err != nil {
		return nil, err
	}
	info, err := file.Stat()
	if err != nil || !info.Mode().IsRegular() {
		return nil, doctorFail("untrusted", "Expected a regular file.")
	}
	if info.Size() > limit {
		return nil, doctorFail("incomplete", "File exceeds the bounded inspection size.")
	}
	if info.Size() > b.remainingBytes {
		return nil, doctorFail("incomplete", "Byte inspection budget exhausted.")
	}
	var result bytes.Buffer
	buffer := make([]byte, 64*1024)
	remaining := limit + 1
	for remaining > 0 {
		if err := b.step(); err != nil {
			return nil, err
		}
		n, readErr := file.Read(buffer[:min(int64(len(buffer)), remaining)])
		b.remainingBytes -= int64(n)
		remaining -= int64(n)
		if b.remainingBytes < 0 {
			return nil, doctorFail("incomplete", "Byte inspection budget exhausted.")
		}
		result.Write(buffer[:n])
		if readErr == io.EOF {
			break
		}
		if readErr != nil {
			return nil, doctorFail("unreadable", "Could not read the inspected file.")
		}
	}
	if int64(result.Len()) > limit {
		return nil, doctorFail("incomplete", "File exceeds the bounded inspection size.")
	}
	return result.Bytes(), nil
}

func doctorPath(path string, directory, protected bool) error {
	if !filepath.IsAbs(path) || filepath.Clean(path) != path || len(path) > 4096 {
		return doctorFail("malformed", "Expected an absolute canonical local path.")
	}
	// Inspect ancestors before the leaf, so no symlinked component is traversed.
	parts := strings.Split(strings.TrimPrefix(path, string(os.PathSeparator)), string(os.PathSeparator))
	current := string(os.PathSeparator)
	paths := []string{current}
	for _, part := range parts {
		if part != "" {
			current = filepath.Join(current, part)
			paths = append(paths, current)
		}
	}
	for index, component := range paths {
		info, err := os.Lstat(component)
		if errors.Is(err, os.ErrNotExist) {
			return doctorFail("missing", "Required local path is absent.")
		}
		if err != nil {
			return doctorFail("unreadable", "Could not inspect the local path.")
		}
		leaf := index == len(paths)-1
		if (!leaf || directory) && !info.IsDir() || leaf && !directory && !info.Mode().IsRegular() {
			return doctorFail("untrusted", "Path contains a symlink, special file, or unexpected file type.")
		}
		if protected {
			if err := validateTrustedMetadata(component, info, !leaf); err != nil {
				return doctorFail("untrusted", "Path fails the supervisor ownership, mode, ACL, or platform protection checks.")
			}
		}
	}
	return nil
}

// Open each directory relative to its already opened parent. Unlike a prior
// lstat/EvalSymlinks check, this also prevents an unprotected corpus ancestor
// being exchanged for a symlink between inspection and open.
func doctorOpen(path string, directory bool) (*os.File, error) {
	flags := unix.O_RDONLY | unix.O_NOFOLLOW | unix.O_NONBLOCK | unix.O_CLOEXEC
	fd, err := unix.Open("/", flags|unix.O_DIRECTORY, 0)
	if err != nil {
		return nil, err
	}
	parts := strings.Split(strings.TrimPrefix(path, "/"), "/")
	for index, part := range parts {
		if part == "" {
			continue
		}
		mode := flags
		if index < len(parts)-1 || directory {
			mode |= unix.O_DIRECTORY
		}
		next, openErr := unix.Openat(fd, part, mode, 0)
		_ = unix.Close(fd)
		if openErr != nil {
			return nil, openErr
		}
		fd = next
	}
	return os.NewFile(uintptr(fd), path), nil
}

func (b *doctorBudget) file(path string, limit int64, executable, protected bool) ([]byte, error) {
	if err := b.step(); err != nil {
		return nil, err
	}
	if err := doctorPath(path, false, protected); err != nil {
		return nil, err
	}
	before, err := os.Lstat(path)
	if err != nil {
		return nil, doctorFail("unreadable", "Could not inspect the local file.")
	}
	if executable && before.Mode().Perm()&0o111 == 0 {
		return nil, doctorFail("malformed", "Required executable has no execute permission.")
	}
	// Nonblocking and no-follow prevent a replaced leaf FIFO/symlink from being
	// opened as data. No data is consumed until the opened inode is rechecked.
	f, err := doctorOpen(path, false)
	if err != nil {
		return nil, doctorFail("unreadable", "Could not open the regular local file.")
	}
	defer f.Close()
	after, err := f.Stat()
	if err != nil || !os.SameFile(before, after) || !after.Mode().IsRegular() {
		return nil, doctorFail("untrusted", "File changed while being inspected.")
	}
	return b.read(f, limit)
}

func doctorJSON(raw []byte, target any) error {
	if err := rejectDuplicateJSONKeys(raw); err != nil {
		return doctorFail("malformed", "JSON contains duplicate keys or is malformed.")
	}
	// encoding/json matches struct fields case-insensitively, unlike the
	// production Python attestation reader. Check the exact JSON names first,
	// including nested typed objects, without accepting case-variant aliases.
	if err := doctorExactFields(raw, reflect.TypeOf(target)); err != nil {
		return err
	}
	decoder := json.NewDecoder(bytes.NewReader(raw))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(target); err != nil {
		return doctorFail("malformed", "JSON does not match the supported public configuration schema.")
	}
	if err := decoder.Decode(new(any)); err != io.EOF {
		return doctorFail("malformed", "Expected exactly one JSON value.")
	}
	return nil
}

func doctorExactFields(raw json.RawMessage, schema reflect.Type) error {
	for schema.Kind() == reflect.Pointer {
		schema = schema.Elem()
	}
	if schema.Kind() != reflect.Struct {
		return nil
	}
	// Decode only schema-defined object levels. In particular, do not expand
	// the corpus content tree into interface{} maps just to check its envelope.
	var object map[string]json.RawMessage
	if err := json.Unmarshal(raw, &object); err != nil || object == nil {
		return doctorFail("malformed", "Expected a public configuration object.")
	}
	fields := make(map[string]reflect.StructField, schema.NumField())
	for index := 0; index < schema.NumField(); index++ {
		field := schema.Field(index)
		fields[strings.Split(field.Tag.Get("json"), ",")[0]] = field
	}
	for name, child := range object {
		field, exists := fields[name]
		if !exists {
			return doctorFail("malformed", "Public JSON field names must match the schema exactly.")
		}
		if err := doctorExactFields(child, field.Type); err != nil {
			return err
		}
	}
	return nil
}

func (r *doctorReport) add(id, owner, action string, check func() (string, error)) {
	detail, err := check()
	status := "observed"
	if err != nil {
		status, detail = "incomplete", "Inspection could not be completed."
		var problem *doctorProblem
		if errors.As(err, &problem) {
			status, detail = problem.status, problem.detail
		}
	}
	r.Checks = append(r.Checks, doctorCheck{id, status, detail, owner, action})
}
func (r *doctorReport) pending(id, owner, detail, action string) {
	r.Checks = append(r.Checks, doctorCheck{id, "not_checked", detail, owner, action})
}

// Read directory batches instead of WalkDir/ReadDir(path), which allocate an
// unbounded directory listing before a visitor can enforce a budget.
func (b *doctorBudget) directory(path string) ([]os.DirEntry, error) {
	if err := b.step(); err != nil {
		return nil, err
	}
	if err := doctorPath(path, true, true); err != nil {
		return nil, err
	}
	f, err := doctorOpen(path, true)
	if err != nil {
		return nil, doctorFail("unreadable", "Could not open protected directory.")
	}
	defer f.Close()
	var entries []os.DirEntry
	for {
		batch, err := f.ReadDir(128)
		for _, entry := range batch {
			if budgetErr := b.step(); budgetErr != nil {
				return nil, budgetErr
			}
			entries = append(entries, entry)
		}
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, doctorFail("unreadable", "Could not enumerate protected directory.")
		}
	}
	sort.Slice(entries, func(i, j int) bool { return entries[i].Name() < entries[j].Name() })
	return entries, nil
}

func (b *doctorBudget) tree(root string, packageHash hash.Hash, relative string, depth int) error {
	if depth > 64 {
		return doctorFail("incomplete", "Directory depth exceeds inspection budget.")
	}
	entries, err := b.directory(root)
	if err != nil {
		return err
	}
	var directories []string
	for _, entry := range entries {
		if err := b.step(); err != nil {
			return err
		}
		path := filepath.Join(root, entry.Name())
		info, err := entry.Info()
		if err != nil {
			return doctorFail("unreadable", "Could not inspect protected tree entry.")
		}
		if !info.IsDir() && !info.Mode().IsRegular() {
			return doctorFail("untrusted", "Protected tree contains a symlink or special file.")
		}
		if err := validateTrustedMetadata(path, info, false); err != nil {
			return doctorFail("untrusted", "Protected tree entry fails ownership, mode, ACL, or platform checks.")
		}
		if info.IsDir() {
			directories = append(directories, entry.Name())
			continue
		}
		// Same startup-carrier set as validateTrustedTree. Full bootstrap/runtime
		// admission remains the production supervisor's responsibility.
		name := strings.ToLower(entry.Name())
		if strings.HasSuffix(name, ".pth") || strings.HasSuffix(name, ".egg-link") || name == "sitecustomize.py" || name == "usercustomize.py" || name == "pyvenv.cfg" || strings.HasPrefix(name, "__editable__") {
			return doctorFail("untrusted", "Protected Python tree contains a forbidden startup or editable injection file.")
		}
		if packageHash != nil {
			raw, err := b.file(path, 32*1024*1024, false, true)
			if err != nil {
				return err
			}
			rel := filepath.ToSlash(filepath.Join(relative, entry.Name()))
			// The provisioner and Python apply path use this exact domain-separated,
			// length-framed, files-before-subdirectories tree identity.
			_ = binary.Write(packageHash, binary.BigEndian, uint64(len([]byte(rel))))
			_, _ = packageHash.Write([]byte(rel))
			_ = binary.Write(packageHash, binary.BigEndian, uint64(len(raw)))
			_, _ = packageHash.Write(raw)
		}
	}
	for _, name := range directories {
		if packageHash != nil && name == "__pycache__" {
			continue
		}
		if err := b.tree(filepath.Join(root, name), packageHash, filepath.Join(relative, name), depth+1); err != nil {
			return err
		}
	}
	return nil
}

type doctorAttestation struct {
	Schema        string `json:"schema"`
	ProvisionedAt string `json:"provisioned_at"`
	Encoder       struct {
		Origin     string `json:"origin_repository"`
		Commit     string `json:"commit"`
		Version    string `json:"version"`
		TreeSHA256 string `json:"package_tree_sha256"`
	} `json:"axiom_encode"`
	Codex *struct {
		Version string `json:"version"`
		SHA256  string `json:"sha256"`
	} `json:"codex_cli,omitempty"`
}

func inspectDoctor(o doctorOptions, b *doctorBudget) doctorReport {
	r := doctorReport{Schema: doctorSchema, Status: "indeterminate", Admission: "not_checked", BuildKind: supervisorBuildKind, Installation: o.installation,
		Checks: []doctorCheck{}, Limits: map[string]int64{"entries_and_read_chunks": 50000, "total_bytes": 512 * 1024 * 1024, "elapsed_work_seconds": 10, "directory_depth": 64}}
	r.ExpectedEncoderCommit, r.ExpectedSupervisorSHA256 = o.expectedCommit, o.expectedSupervisor
	if o.corpusRoot != "" {
		r.CorpusSelection = &doctorCorpusSelection{o.corpusRoot, o.releaseName, o.releaseDigest, filepath.Join(o.corpusRoot, "releases", o.releaseName, o.releaseDigest+".json")}
	}
	installAction := "Have the runtime operator inspect/re-provision the approved installation using the documented provisioner; do not automate sudo."
	r.add("process_identity", doctorInstallOwner, "Run the diagnostic as the unprivileged runtime operator.", func() (string, error) {
		if supervisorBuildKind != "production" {
			return "", doctorFail("untrusted", "This is a nonpublishable test-fixture diagnostic build.")
		}
		if err := validateTrustProcessIdentity(); err != nil {
			return "", doctorFail("untrusted", "Invoking process fails the production unprivileged identity checks.")
		}
		return "Unprivileged process identity checks passed; diagnostic output is not a trust attestation.", nil
	})
	r.add("environment", doctorInstallOwner, "Remove forbidden public/private signing environment inputs. For subscription launch, pass auth/outbox explicitly and unset CODEX_HOME; the doctor reads no credential values.", func() (string, error) {
		for name := range privateEnvironmentNames {
			if _, exists := os.LookupEnv(name); exists {
				return "", doctorFail("malformed", "A forbidden private signing environment variable is present (value not inspected).")
			}
		}
		for name := range publicEnvironmentNames {
			if _, exists := os.LookupEnv(name); exists {
				return "", doctorFail("malformed", "A forbidden public-root environment variable is present (value not inspected).")
			}
		}
		return "No forbidden signing environment names are set. Ambient CODEX_HOME is not a supervised subscription configuration.", nil
	})
	r.add("subscription_environment", "subscription lane operator", "Unset ambient CODEX_HOME for the actual supervised launch and pass --codex-subscription-auth / --codex-auth-outbox explicitly. The doctor does not change the environment or inspect credentials.", func() (string, error) {
		if value, present := os.LookupEnv("CODEX_HOME"); present && value != "" {
			return "", doctorFail("malformed", "Ambient CODEX_HOME is set and would be refused by a subscription supervisor invocation (value suppressed).")
		}
		return "Ambient CODEX_HOME does not conflict with the explicit supervised subscription custody path.", nil
	})
	r.add("supervisor", doctorInstallOwner, installAction, func() (string, error) {
		raw, err := b.file(filepath.Join(o.installation, "axiom-encode-signing-supervisor"), 64*1024*1024, true, true)
		if err != nil {
			return "", err
		}
		if !isNativeExecutableHeader(raw) {
			return "", doctorFail("malformed", "Supervisor is not a supported native executable.")
		}
		// Never parse ELF/Mach-O/Go metadata here. Even a small input can make
		// general binary readers allocate/decompress far beyond our byte budget.
		// Header and hash observations do not establish format validity or kind.
		digest := sha256.Sum256(raw)
		if o.expectedSupervisor != "" && hex.EncodeToString(digest[:]) != o.expectedSupervisor {
			return "", doctorFail("stale", "Supervisor bytes differ from the operator-selected expected SHA-256.")
		}
		return fmt.Sprintf("Protected file has a native executable header; sha256=%x. Format validity, build kind and provenance are not established; binary was not executed or parsed.", digest), nil
	})
	if o.expectedSupervisor == "" {
		r.pending("supervisor_provenance", doctorInstallOwner, "No operator-selected expected supervisor hash supplied; executable header and file protection do not establish production build kind or provenance.", "Compare with the approved production build artifact and pass --expected-supervisor-sha256. The caller's expected hash is not itself authenticated provenance.")
	}
	runtimeRoot := filepath.Join(o.installation, "python")
	packageRoot := ""
	r.add("python_launcher", doctorInstallOwner, installAction, func() (string, error) {
		raw, err := b.file(filepath.Join(o.installation, "axiom-encode"), 4096, true, true)
		if err != nil {
			return "", err
		}
		interpreter, arg, script, err := parseShebang(raw)
		if err != nil || !script || arg != "-I" || !isPythonInterpreterName(filepath.Base(interpreter)) || filepath.Dir(interpreter) != filepath.Join(runtimeRoot, "bin") || !strings.HasPrefix(filepath.Base(interpreter), "python3.") {
			return "", doctorFail("malformed", "Launcher must declare the provisioner's native versioned Python interpreter under installation/python/bin with exactly -I.")
		}
		native, err := b.file(interpreter, 64*1024*1024, true, true)
		if err != nil {
			return "", err
		}
		if !isNativeExecutableHeader(native) {
			return "", doctorFail("malformed", "Python interpreter is not a supported native executable.")
		}
		packageRoot = filepath.Join(runtimeRoot, "lib", filepath.Base(interpreter), "site-packages", "axiom_encode")
		return "Protected isolated shebang and native interpreter observed. Launcher body is intentionally never executed by the supervisor.", nil
	})
	r.add("python_runtime_tree", doctorInstallOwner, installAction, func() (string, error) {
		if err := b.tree(runtimeRoot, nil, "", 0); err != nil {
			return "", err
		}
		return "Bounded tree inspection passed filesystem/startup-carrier checks; dynamic loader and bootstrap admission were not run.", nil
	})
	r.add("runtime_git_wrapper", doctorInstallOwner, installAction, func() (string, error) {
		raw, err := b.file(filepath.Join(runtimeRoot, "bin", "git"), 64*1024, true, true)
		if err != nil {
			return "", err
		}
		interpreter, arg, script, err := parseShebang(raw)
		if err != nil || !script || arg != "-I" || filepath.Dir(interpreter) != filepath.Join(runtimeRoot, "bin") || !isPythonInterpreterName(filepath.Base(interpreter)) {
			return "", doctorFail("malformed", "Provisioned Git wrapper must name an isolated protected runtime Python interpreter.")
		}
		native, err := b.file(interpreter, 64*1024*1024, true, true)
		if err != nil {
			return "", err
		}
		if !isNativeExecutableHeader(native) {
			return "", doctorFail("malformed", "Git wrapper interpreter is not a supported native executable.")
		}
		return "Protected isolated Git wrapper is present. Its delegated Git binary, behavior and provenance are not executed or verified.", nil
	})
	var attestation doctorAttestation
	var attestationOK bool
	r.add("runtime_attestation", doctorInstallOwner, installAction, func() (string, error) {
		raw, err := b.file(filepath.Join(runtimeRoot, "runtime-attestation.json"), 64*1024, false, true)
		if err != nil {
			return "", err
		}
		if err := doctorJSON(raw, &attestation); err != nil {
			return "", err
		}
		if attestation.Schema != "axiom-encode/trusted-runtime-attestation/v1" || attestation.Encoder.Origin != "github.com/TheAxiomFoundation/axiom-encode" || !doctorCommit.MatchString(attestation.Encoder.Commit) || !doctorVersion.MatchString(attestation.Encoder.Version) || !doctorDigest.MatchString(attestation.Encoder.TreeSHA256) {
			return "", doctorFail("malformed", "Runtime attestation has unsupported or invalid encoder identity fields.")
		}
		if _, err := time.Parse(time.RFC3339Nano, attestation.ProvisionedAt); err != nil {
			return "", doctorFail("malformed", "Runtime attestation has no valid provisioned_at timestamp.")
		}
		if attestation.Codex != nil && (attestation.Codex.Version == "" || !doctorDigest.MatchString(attestation.Codex.SHA256)) {
			return "", doctorFail("malformed", "Runtime attestation has invalid Codex identity fields.")
		}
		attestationOK = true
		if o.expectedCommit != "" && o.expectedCommit != attestation.Encoder.Commit {
			return "", doctorFail("stale", "Protected runtime attests encoder "+attestation.Encoder.Version+" at "+attestation.Encoder.Commit+"; differs from the requested commit.")
		}
		return "Protected runtime attests encoder " + attestation.Encoder.Version + " at " + attestation.Encoder.Commit + ". Package bytes are checked separately.", nil
	})
	if o.expectedCommit == "" {
		r.pending("encoder_revision", doctorInstallOwner, "No expected encoder commit supplied; freshness is unknown.", "Pass --expected-encoder-commit using the approved immutable encoder revision.")
	}
	r.add("encoder_package", doctorInstallOwner, installAction, func() (string, error) {
		if packageRoot == "" || !attestationOK {
			return "", doctorFail("incomplete", "Package identity comparison requires a valid protected launcher and attestation.")
		}
		for _, name := range []string{"__init__.py", "_trusted_signing_bootstrap.py"} {
			if _, err := b.file(filepath.Join(packageRoot, name), 2*1024*1024, false, true); err != nil {
				return "", err
			}
		}
		digest := sha256.New()
		_, _ = digest.Write([]byte("axiom-eval-tree-v1\x00"))
		if err := b.tree(packageRoot, digest, "", 0); err != nil {
			return "", err
		}
		if hex.EncodeToString(digest.Sum(nil)) != attestation.Encoder.TreeSHA256 {
			return "", doctorFail("untrusted", "Installed encoder package bytes do not match the protected runtime attestation.")
		}
		return "Installed package tree matches the protected attestation. This does not execute or admit the Python runtime.", nil
	})
	r.add("public_trust_roots", doctorInstallOwner, "Have the trust-root custodian supply the approved protected v2/v3 public-root configuration, including required retired verification keys; do not generate substitute roots.", func() (string, error) {
		raw, err := b.file(filepath.Join(o.installation, "signing-trust-roots.json"), 64*1024, false, true)
		if err != nil {
			return "", err
		}
		if err := rejectDuplicateJSONKeys(raw); err != nil {
			return "", doctorFail("malformed", "Public trust-root JSON is malformed or contains duplicate keys.")
		}
		_, _, ring, err := parseProtectedTrustRoots(raw, "<public configuration>")
		if err != nil {
			return "", doctorFail("malformed", "Public trust roots fail the production schema/key-separation parser (values suppressed).")
		}
		return fmt.Sprintf("Protected public configuration passes the production parser with %d corpus verification key(s). Custodian approval and release signatures are not checked.", len(ring)), nil
	})
	r.add("subscription_codex", doctorInstallOwner, "Have the operator install the approved pinned Codex CLI via --install-pinned-codex-cli; refresh pin/config/attestation together through the existing process.", func() (string, error) {
		raw, err := b.file(filepath.Join(o.installation, "codex-cli.json"), 16*1024, false, true)
		if err != nil {
			return "", err
		}
		var config trustedCodexCLI
		if err := doctorJSON(raw, &config); err != nil {
			return "", err
		}
		if config.Schema != "axiom-encode/trusted-codex-cli/v1" || !doctorVersion.MatchString(config.Version) || !doctorDigest.MatchString(config.SHA256) || config.Path != filepath.Join(o.installation, "bin", "codex") {
			return "", doctorFail("malformed", "Codex config must identify the provisioner's installation/bin/codex with a version and SHA-256.")
		}
		executable, err := b.file(config.Path, 256*1024*1024, true, true)
		if err != nil {
			return "", err
		}
		if !isNativeExecutableHeader(executable) {
			return "", doctorFail("malformed", "Pinned Codex CLI is not a supported native executable.")
		}
		digest := sha256.Sum256(executable)
		if hex.EncodeToString(digest[:]) != config.SHA256 {
			return "", doctorFail("untrusted", "Pinned Codex CLI does not match the protected config SHA-256.")
		}
		if !attestationOK || attestation.Codex == nil {
			return "", doctorFail("incomplete", "Pinned executable hash matches config, but protected runtime Codex attestation is unavailable.")
		}
		if attestation.Codex.Version != config.Version || attestation.Codex.SHA256 != config.SHA256 {
			return "", doctorFail("untrusted", "Codex config and protected runtime attestation disagree.")
		}
		return "Protected Codex CLI " + config.Version + " hash matches configuration and runtime attestation; executable and subscription were not invoked.", nil
	})
	r.pending("subscription_credentials", "subscription lane operator", "Credential contents, validity, refresh outbox, and subscription capacity were not inspected.", "Use the documented --codex-subscription-auth and --codex-auth-outbox custody path on the existing subscription lane, with ambient CODEX_HOME unset. No API key or reset is required by this diagnostic.")
	if o.corpusRoot == "" {
		r.pending("corpus_release", doctorCorpusOwner, "No immutable local corpus release selected.", "Pass --corpus-root, --release-name and --release-content-sha256 from the target's checked-in .axiom/toolchain.toml.")
	} else {
		r.add("corpus_release", doctorCorpusOwner, "Materialize the selected signed release and artifacts using the corpus owner's existing release process. Candidate sources need separate review/admission; never change serving pointers to fix readiness.", func() (string, error) {
			path := filepath.Join(o.corpusRoot, "releases", o.releaseName, o.releaseDigest+".json")
			raw, err := b.file(path, 8*1024*1024, false, false)
			if err != nil {
				return "", err
			}
			var payload struct {
				Schema    string                     `json:"schema_version"`
				Release   string                     `json:"release"`
				Digest    string                     `json:"content_sha256"`
				Content   map[string]json.RawMessage `json:"content"`
				Signature json.RawMessage            `json:"signature"`
			}
			if err := doctorJSON(raw, &payload); err != nil {
				return "", err
			}
			if (payload.Schema != "axiom-corpus/release-object/v2" && payload.Schema != "axiom-corpus/release-object/v3") || payload.Release != o.releaseName || payload.Digest != o.releaseDigest || len(payload.Content) == 0 {
				return "", doctorFail("malformed", "Release envelope schema or claimed immutable identity does not match the selected release.")
			}
			return "Selected release object is locally present with matching envelope identity only. Content digest, signature, source coverage and referenced artifacts are NOT verified.", nil
		})
	}
	r.pending("release_admission", doctorCorpusOwner, "No corpus release is admitted by this diagnostic, even if its object is present.", "The real protected broker and corpus resolver must verify the signature, content identity, selected source scope and artifact bytes.")
	r.pending("apply_signer", "trusted signing/admission owner (encode#1192; production-signing reviewers)", "No signer descriptor is accepted or contacted. The production apply-signer launcher is CI-bound; subscription installation alone does not supply local signed apply.", "Obtain the existing authorized external signer/admission process from its owner. Do not use --allow-local-dev, fake CI identity, replacement roots or protected signing for a readiness check.")
	r.pending("runtime_admission", doctorInstallOwner, "No supervisor bootstrap, dynamic loader, broker peer authentication or required runtime subprocess tools were executed.", "After blockers are resolved, the owner must use the actual supervised invocation; observations here never authorize generation, signing or serving.")
	for _, check := range r.Checks {
		if check.Status != "observed" && check.Status != "not_checked" && check.Status != "incomplete" {
			r.Status = "blocked"
		}
	}
	return r
}

func runDoctor(args []string, stdout, stderr io.Writer) int {
	flags := flag.NewFlagSet("axiom-encode-signing-supervisor --doctor", flag.ContinueOnError)
	// Do not echo arbitrary flag values (which could contain pasted credentials).
	flags.SetOutput(io.Discard)
	o := doctorOptions{}
	flags.StringVar(&o.installation, "installation", "/opt/axiom-verification", "provisioner installation root")
	flags.StringVar(&o.expectedCommit, "expected-encoder-commit", "", "approved immutable encoder commit")
	flags.StringVar(&o.expectedSupervisor, "expected-supervisor-sha256", "", "approved supervisor artifact SHA-256")
	flags.StringVar(&o.corpusRoot, "corpus-root", "", "local corpus checkout root")
	flags.StringVar(&o.releaseName, "release-name", "", "immutable release name from toolchain.toml")
	flags.StringVar(&o.releaseDigest, "release-content-sha256", "", "immutable release content SHA-256 from toolchain.toml")
	flags.BoolVar(&o.json, "json", false, "structured report on stdout")
	if err := flags.Parse(args); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			if _, err := fmt.Fprintln(stdout, "Offline, nonmutating runtime observations; never admission. Does not execute binaries, read auth, connect, sign or provision.\nUsage: axiom-encode-signing-supervisor --doctor [--installation /opt/axiom-verification] [--expected-encoder-commit COMMIT] [--expected-supervisor-sha256 SHA256] [--corpus-root PATH --release-name NAME --release-content-sha256 SHA256] [--json]\nExit: 1 known blockers; 2 indeterminate (admission still unchecked); 64 invalid arguments/output errors; 0 help."); err != nil {
				return 64
			}
			return 0
		}
		fmt.Fprintln(stderr, "doctor: invalid arguments; use --doctor --help (values suppressed)")
		return 64
	}
	validPath := func(path string) bool {
		return filepath.IsAbs(path) && filepath.Clean(path) == path && len(path) <= 4096 && !strings.ContainsAny(path, "\x00\r\n\t")
	}
	if len(flags.Args()) != 0 || !validPath(o.installation) || (o.expectedCommit != "" && !doctorCommit.MatchString(o.expectedCommit)) || (o.expectedSupervisor != "" && !doctorDigest.MatchString(o.expectedSupervisor)) || ((o.corpusRoot != "" || o.releaseName != "" || o.releaseDigest != "") && (!validPath(o.corpusRoot) || len(o.releaseName) > 255 || !doctorRelease.MatchString(o.releaseName) || !doctorDigest.MatchString(o.releaseDigest))) {
		fmt.Fprintln(stderr, "doctor: invalid path, expected identity, or incomplete release selection; use --doctor --help (values suppressed)")
		return 64
	}
	report := inspectDoctor(o, newDoctorBudget())
	if o.json {
		encoder := json.NewEncoder(stdout)
		encoder.SetIndent("", "  ")
		if err := encoder.Encode(report); err != nil {
			return 64
		}
	} else {
		var rendered bytes.Buffer
		fmt.Fprintf(&rendered, "Runtime readiness: %s; admission NOT CHECKED\n", report.Status)
		for _, check := range report.Checks {
			fmt.Fprintf(&rendered, "[%s] %s: %s\n", check.Status, check.ID, check.Detail)
			if check.Status != "observed" {
				fmt.Fprintf(&rendered, "  Owner: %s\n  Next: %s\n", check.Owner, check.Action)
			}
		}
		if _, err := io.Copy(stdout, &rendered); err != nil {
			return 64
		}
	}
	if report.Status == "blocked" {
		return 1
	}
	return 2
}
