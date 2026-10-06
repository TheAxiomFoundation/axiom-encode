//go:build darwin || linux

package main

import (
	"bytes"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"syscall"
)

const (
	maxExecutableHeaderBytes = 4 * 1024
	maxInterpreterDepth      = 4
	accessWriteMode          = 2
)

// trustedExecutionChain is the immutable execution surface the supervisor has
// validated before it delegates a signing capability. A Python command must
// use an isolated, absolute interpreter and explicitly declared runtime,
// import, package, and bootstrap paths. Native supervised commands are rejected.
type trustedExecutionChain struct {
	Executable         string
	Interpreters       []string
	PythonInterpreter  string
	PythonBootstrap    string
	PythonRuntimeRoots []string
	PythonImportRoots  []string
	PythonPackageRoot  string
}

// validateTrustedExecutionChain validates an axiom-encode executable and all
// code locations deterministically reachable from its shebang. Python package
// discovery is intentionally forbidden: callers must declare the hermetic
// runtime roots and exact axiom_encode package directory they provisioned.
func validateTrustedExecutionChain(
	executable string,
	pythonRuntimeRoots []string,
	pythonImportRoots []string,
	pythonPackageRoot string,
) (trustedExecutionChain, error) {
	chain := trustedExecutionChain{}
	if err := validateTrustProcessIdentity(); err != nil {
		return chain, err
	}
	trustedExecutable, header, err := inspectTrustedExecutable(executable)
	if err != nil {
		return chain, err
	}
	chain.Executable = trustedExecutable

	currentPath := trustedExecutable
	currentHeader := header
	seen := map[string]struct{}{trustedExecutable: {}}
	var interpreterArgument string
	for depth := 0; ; depth++ {
		interpreter, argument, script, err := parseShebang(currentHeader)
		if err != nil {
			return trustedExecutionChain{}, fmt.Errorf(
				"could not validate interpreter for %s: %w", currentPath, err,
			)
		}
		if !script {
			if !isNativeExecutableHeader(currentHeader) {
				return trustedExecutionChain{}, fmt.Errorf(
					"trusted executable is neither an isolated Python script nor a supported native binary: %s",
					currentPath,
				)
			}
			break
		}
		if depth > 0 {
			return trustedExecutionChain{}, errors.New(
				"nested shebang interpreters are forbidden; axiom-encode must name its native Python interpreter directly",
			)
		}
		if depth >= maxInterpreterDepth {
			return trustedExecutionChain{}, fmt.Errorf(
				"interpreter chain exceeds %d levels", maxInterpreterDepth,
			)
		}
		if _, duplicate := seen[interpreter]; duplicate {
			return trustedExecutionChain{}, fmt.Errorf(
				"interpreter chain contains a cycle at %s", interpreter,
			)
		}
		if depth == 0 {
			interpreterArgument = argument
		} else if argument != "" {
			return trustedExecutionChain{}, errors.New(
				"nested interpreter scripts must not add shebang arguments",
			)
		}
		trustedInterpreter, nextHeader, err := inspectTrustedExecutable(interpreter)
		if err != nil {
			return trustedExecutionChain{}, fmt.Errorf(
				"untrusted shebang interpreter %s: %w", interpreter, err,
			)
		}
		chain.Interpreters = append(chain.Interpreters, trustedInterpreter)
		seen[trustedInterpreter] = struct{}{}
		currentPath = trustedInterpreter
		currentHeader = nextHeader
	}

	if len(chain.Interpreters) == 0 {
		return trustedExecutionChain{}, errors.New(
			"supervised axiom-encode must be the protected Python launcher; native command execution has no admitted signing contract",
		)
	}

	terminalInterpreter := chain.Interpreters[len(chain.Interpreters)-1]
	if !isPythonInterpreterName(filepath.Base(terminalInterpreter)) {
		return trustedExecutionChain{}, fmt.Errorf(
			"scripted axiom-encode must terminate in a Python interpreter, got %s",
			terminalInterpreter,
		)
	}
	if interpreterArgument != "-I" {
		return trustedExecutionChain{}, errors.New(
			"axiom-encode Python shebang must contain exactly the isolated-mode argument -I",
		)
	}
	if len(pythonRuntimeRoots) != 1 {
		return trustedExecutionChain{}, errors.New(
			"Python axiom-encode requires exactly one self-contained trusted runtime root",
		)
	}
	if len(pythonImportRoots) == 0 {
		return trustedExecutionChain{}, errors.New(
			"Python axiom-encode requires at least one explicit trusted import root",
		)
	}
	if pythonPackageRoot == "" {
		return trustedExecutionChain{}, errors.New(
			"Python axiom-encode requires an exact trusted axiom_encode package root",
		)
	}

	canonicalRuntimeRoots, err := validateTrustedTreeSet(pythonRuntimeRoots)
	if err != nil {
		return trustedExecutionChain{}, fmt.Errorf("untrusted Python runtime: %w", err)
	}
	canonicalPackageRoot, err := validateTrustedTree(pythonPackageRoot)
	if err != nil {
		return trustedExecutionChain{}, fmt.Errorf("untrusted axiom_encode package: %w", err)
	}
	if filepath.Base(canonicalPackageRoot) != "axiom_encode" {
		return trustedExecutionChain{}, errors.New(
			"trusted Python package root must name the exact axiom_encode directory",
		)
	}
	packageInitializer := filepath.Join(canonicalPackageRoot, "__init__.py")
	_, initializer, err := inspectTrustedRegularFile(packageInitializer, false)
	if err != nil {
		return trustedExecutionChain{}, fmt.Errorf(
			"trusted axiom_encode package has no protected __init__.py: %w", err,
		)
	}
	if err := initializer.Close(); err != nil {
		return trustedExecutionChain{}, fmt.Errorf(
			"could not close trusted axiom_encode package initializer: %w", err,
		)
	}
	if !containedByAny(terminalInterpreter, canonicalRuntimeRoots) {
		return trustedExecutionChain{}, errors.New(
			"Python interpreter must be contained by a declared trusted runtime root",
		)
	}
	if !containedByAny(canonicalPackageRoot, canonicalRuntimeRoots) {
		return trustedExecutionChain{}, errors.New(
			"axiom_encode package must be contained by a declared trusted runtime root",
		)
	}
	canonicalImportRoots, err := validateTrustedTreeSet(pythonImportRoots)
	if err != nil {
		return trustedExecutionChain{}, fmt.Errorf("untrusted Python import roots: %w", err)
	}
	for _, importRoot := range canonicalImportRoots {
		if !pathContains(canonicalRuntimeRoots[0], importRoot) {
			return trustedExecutionChain{}, errors.New(
				"every Python import root must be contained by the self-contained runtime",
			)
		}
	}
	packageParent := filepath.Dir(canonicalPackageRoot)
	packageParentDeclared := false
	for _, importRoot := range canonicalImportRoots {
		if importRoot == packageParent {
			packageParentDeclared = true
			break
		}
	}
	if !packageParentDeclared {
		return trustedExecutionChain{}, errors.New(
			"the exact parent of the axiom_encode package must be an explicit import root",
		)
	}
	bootstrap := filepath.Join(canonicalPackageRoot, "_trusted_signing_bootstrap.py")
	_, bootstrapFile, err := inspectTrustedRegularFile(bootstrap, false)
	if err != nil {
		return trustedExecutionChain{}, fmt.Errorf(
			"trusted Python bootstrap is not protected: %w", err,
		)
	}
	if err := bootstrapFile.Close(); err != nil {
		return trustedExecutionChain{}, fmt.Errorf(
			"could not close trusted Python bootstrap: %w", err,
		)
	}

	chain.Executable = terminalInterpreter
	chain.PythonInterpreter = terminalInterpreter
	chain.PythonBootstrap = bootstrap
	chain.PythonRuntimeRoots = canonicalRuntimeRoots
	chain.PythonImportRoots = canonicalImportRoots
	chain.PythonPackageRoot = canonicalPackageRoot
	return chain, nil
}

// validateTrustedNativeExecutable validates the compiled supervisor/broker
// without introducing a mutable path re-exec.
func validateTrustedNativeExecutable(path string) (string, error) {
	if err := validateTrustProcessIdentity(); err != nil {
		return "", err
	}
	trustedPath, header, err := inspectTrustedExecutable(path)
	if err != nil {
		return "", err
	}
	if !isNativeExecutableHeader(header) {
		return "", errors.New("trusted supervisor must be a native ELF or Mach-O executable")
	}
	return trustedPath, nil
}

func validateTrustProcessIdentity() error {
	if err := validatePlatformPrivilegeState(); err != nil {
		return err
	}
	if trustPolicyAllowsWritablePath() {
		return nil
	}
	if os.Getuid() != os.Geteuid() {
		return errors.New("signing supervisor must not run setuid or with mismatched user identities")
	}
	if os.Geteuid() == 0 {
		return errors.New("signing supervisor must run as an unprivileged user")
	}
	return nil
}

func inspectTrustedExecutable(path string) (string, []byte, error) {
	trustedPath, file, err := inspectTrustedRegularFile(path, true)
	if err != nil {
		return "", nil, err
	}
	defer file.Close()
	header := make([]byte, maxExecutableHeaderBytes)
	count, readErr := io.ReadFull(file, header)
	if readErr != nil && !errors.Is(readErr, io.ErrUnexpectedEOF) {
		return "", nil, fmt.Errorf("could not inspect executable header: %w", readErr)
	}
	if count == 0 {
		return "", nil, errors.New("trusted executable must not be empty")
	}
	return trustedPath, header[:count], nil
}

func inspectTrustedRegularFile(
	path string,
	requireExecutable bool,
) (string, *os.File, error) {
	canonical, err := validateCanonicalAbsolutePath(path)
	if err != nil {
		return "", nil, err
	}
	if err := validateTrustedAncestors(filepath.Dir(canonical)); err != nil {
		return "", nil, err
	}
	lstat, err := os.Lstat(canonical)
	if err != nil {
		return "", nil, fmt.Errorf("could not inspect trusted file %s: %w", canonical, err)
	}
	if lstat.Mode()&os.ModeSymlink != 0 || !lstat.Mode().IsRegular() {
		return "", nil, fmt.Errorf("trusted path must be a regular non-symlink file: %s", canonical)
	}
	if requireExecutable && lstat.Mode().Perm()&0o111 == 0 {
		return "", nil, fmt.Errorf("trusted file is not executable: %s", canonical)
	}
	if err := validateTrustedMetadata(canonical, lstat, false); err != nil {
		return "", nil, err
	}
	file, err := os.Open(canonical)
	if err != nil {
		return "", nil, fmt.Errorf("could not open trusted file %s: %w", canonical, err)
	}
	fstat, err := file.Stat()
	if err != nil {
		file.Close()
		return "", nil, fmt.Errorf("could not re-inspect trusted file %s: %w", canonical, err)
	}
	if !os.SameFile(lstat, fstat) {
		file.Close()
		return "", nil, fmt.Errorf("trusted file changed while it was inspected: %s", canonical)
	}
	return canonical, file, nil
}

func validateTrustedTreeSet(roots []string) ([]string, error) {
	canonical := make([]string, 0, len(roots))
	seen := make(map[string]struct{}, len(roots))
	for _, root := range roots {
		trustedRoot, err := validateTrustedTree(root)
		if err != nil {
			return nil, err
		}
		if _, duplicate := seen[trustedRoot]; duplicate {
			return nil, fmt.Errorf("trusted runtime root is repeated: %s", trustedRoot)
		}
		for _, other := range canonical {
			if pathContains(other, trustedRoot) || pathContains(trustedRoot, other) {
				return nil, fmt.Errorf(
					"trusted runtime roots must not overlap: %s and %s", other, trustedRoot,
				)
			}
		}
		seen[trustedRoot] = struct{}{}
		canonical = append(canonical, trustedRoot)
	}
	return canonical, nil
}

func validateTrustedTree(root string) (string, error) {
	canonical, err := validateCanonicalAbsolutePath(root)
	if err != nil {
		return "", err
	}
	if err := validateTrustedAncestors(filepath.Dir(canonical)); err != nil {
		return "", err
	}
	rootInfo, err := os.Lstat(canonical)
	if err != nil {
		return "", fmt.Errorf("could not inspect trusted tree %s: %w", canonical, err)
	}
	if rootInfo.Mode()&os.ModeSymlink != 0 || !rootInfo.IsDir() {
		return "", fmt.Errorf("trusted tree root must be a real directory: %s", canonical)
	}
	err = filepath.WalkDir(canonical, func(path string, entry os.DirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		info, err := entry.Info()
		if err != nil {
			return err
		}
		if info.Mode()&os.ModeSymlink != 0 {
			return fmt.Errorf("trusted tree must not contain symlinks: %s", path)
		}
		if !info.IsDir() && !info.Mode().IsRegular() {
			return fmt.Errorf("trusted tree contains a special file: %s", path)
		}
		if info.Mode().IsRegular() {
			name := strings.ToLower(entry.Name())
			if strings.HasSuffix(name, ".pth") ||
				strings.HasSuffix(name, ".egg-link") ||
				name == "sitecustomize.py" ||
				name == "usercustomize.py" ||
				name == "pyvenv.cfg" ||
				strings.HasPrefix(name, "__editable__") {
				return fmt.Errorf(
					"trusted Python tree contains forbidden startup or editable injection: %s",
					path,
				)
			}
		}
		return validateTrustedMetadata(path, info, false)
	})
	if err != nil {
		return "", fmt.Errorf("could not validate trusted tree %s: %w", canonical, err)
	}
	return canonical, nil
}

func validateTrustedAncestors(directory string) error {
	for {
		info, err := os.Lstat(directory)
		if err != nil {
			return fmt.Errorf("could not inspect trusted ancestor %s: %w", directory, err)
		}
		if info.Mode()&os.ModeSymlink != 0 || !info.IsDir() {
			return fmt.Errorf("trusted ancestor must be a real directory: %s", directory)
		}
		if err := validateTrustedMetadata(directory, info, true); err != nil {
			return err
		}
		parent := filepath.Dir(directory)
		if parent == directory {
			return nil
		}
		directory = parent
	}
}

func validateTrustedMetadata(path string, info os.FileInfo, ancestor bool) error {
	stat, ok := info.Sys().(*syscall.Stat_t)
	if !ok {
		return fmt.Errorf("could not determine trusted path owner: %s", path)
	}
	if !trustPolicyAcceptsOwner(stat.Uid, uint32(os.Geteuid())) {
		return fmt.Errorf("trusted path must be %s: %s", trustPolicyDescription, path)
	}
	if err := validatePlatformPathSecurity(path); err != nil {
		return err
	}
	if ancestor &&
		info.Mode()&os.ModeSticky != 0 &&
		info.Mode()&(os.ModeSetuid|os.ModeSetgid) == 0 &&
		trustPolicyAllowsRootOwnedStickyAncestor(stat.Uid) {
		return nil
	}
	if info.Mode().Perm()&0o022 != 0 {
		return fmt.Errorf("trusted path must not be group- or other-writable: %s", path)
	}
	if info.Mode()&(os.ModeSetuid|os.ModeSetgid|os.ModeSticky) != 0 {
		return fmt.Errorf("trusted path must not carry setuid, setgid, or sticky mode bits: %s", path)
	}
	if !trustPolicyAllowsWritablePath() && syscall.Access(path, accessWriteMode) == nil {
		return fmt.Errorf("trusted path is writable by the invoking user or an ACL: %s", path)
	}
	return nil
}

func validateCanonicalAbsolutePath(path string) (string, error) {
	if path == "" || !filepath.IsAbs(path) {
		return "", errors.New("trusted path must be absolute")
	}
	if filepath.Clean(path) != path {
		return "", fmt.Errorf("trusted path must already be canonical: %s", path)
	}
	resolved, err := filepath.EvalSymlinks(path)
	if err != nil {
		return "", fmt.Errorf("could not resolve trusted path %s: %w", path, err)
	}
	if resolved != path {
		return "", fmt.Errorf("trusted path must not contain symlinks: %s", path)
	}
	return path, nil
}

func parseShebang(header []byte) (string, string, bool, error) {
	if !bytes.HasPrefix(header, []byte("#!")) {
		return "", "", false, nil
	}
	lineEnd := bytes.IndexByte(header, '\n')
	if lineEnd < 0 {
		return "", "", true, errors.New("shebang line is not terminated")
	}
	line := header[2:lineEnd]
	if bytes.IndexByte(line, '\r') >= 0 || bytes.IndexByte(line, 0) >= 0 {
		return "", "", true, errors.New("shebang contains forbidden control bytes")
	}
	fields := bytes.Fields(line)
	if len(fields) == 0 || len(fields) > 2 {
		return "", "", true, errors.New(
			"shebang must contain one absolute interpreter and at most one argument",
		)
	}
	interpreter := string(fields[0])
	if !filepath.IsAbs(interpreter) || filepath.Clean(interpreter) != interpreter {
		return "", "", true, errors.New("shebang interpreter must be an absolute canonical path")
	}
	if filepath.Base(interpreter) == "env" {
		return "", "", true, errors.New("environment-selected shebang interpreters are forbidden")
	}
	argument := ""
	if len(fields) == 2 {
		argument = string(fields[1])
	}
	return interpreter, argument, true, nil
}

func isPythonInterpreterName(name string) bool {
	if name == "python" || name == "python3" {
		return true
	}
	if !strings.HasPrefix(name, "python3.") {
		return false
	}
	version := strings.TrimPrefix(name, "python3.")
	version = strings.TrimSuffix(version, "t")
	if version == "" {
		return false
	}
	for _, character := range version {
		if character < '0' || character > '9' {
			return false
		}
	}
	return true
}

func isNativeExecutableHeader(header []byte) bool {
	if len(header) < 4 {
		return false
	}
	if bytes.Equal(header[:4], []byte{0x7f, 'E', 'L', 'F'}) {
		return true
	}
	// Mach-O 32/64-bit and universal/fat binaries, in either byte order.
	for _, magic := range [][4]byte{
		{0xfe, 0xed, 0xfa, 0xce},
		{0xce, 0xfa, 0xed, 0xfe},
		{0xfe, 0xed, 0xfa, 0xcf},
		{0xcf, 0xfa, 0xed, 0xfe},
		{0xca, 0xfe, 0xba, 0xbe},
		{0xbe, 0xba, 0xfe, 0xca},
		{0xca, 0xfe, 0xba, 0xbf},
		{0xbf, 0xba, 0xfe, 0xca},
	} {
		if bytes.Equal(header[:4], magic[:]) {
			return true
		}
	}
	return false
}

func containedByAny(path string, roots []string) bool {
	for _, root := range roots {
		if pathContains(root, path) {
			return true
		}
	}
	return false
}

func pathContains(root, path string) bool {
	relative, err := filepath.Rel(root, path)
	if err != nil {
		return false
	}
	return relative == "." || (relative != ".." && !strings.HasPrefix(relative, ".."+string(os.PathSeparator)))
}
