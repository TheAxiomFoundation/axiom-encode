//go:build (darwin || linux) && signing_supervisor_test_fixture

package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// newFixtureRuntimeTree builds a minimal current-user-owned runtime tree that
// the fixture trust policy accepts before anything is planted in it. It
// carries names a real CPython prefix or dependency holds that sit close to a
// startup carrier, so every test that plants into it also proves they stay
// accepted.
func newFixtureRuntimeTree(t *testing.T) string {
	t.Helper()
	temporary, err := filepath.EvalSymlinks(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	root := filepath.Join(temporary, "runtime")
	for _, directory := range []string{
		"bin",
		"include/python3.13",
		"lib/python3.13/config-3.13-darwin",
		"lib/python3.13/encodings",
		"lib/python3.13/zipfile",
		"lib/python3.13/site-packages/axiom_encode",
		"lib/python3.13/site-packages/pkg",
	} {
		if err := os.MkdirAll(filepath.Join(root, directory), 0o700); err != nil {
			t.Fatal(err)
		}
	}
	for _, path := range []string{
		"bin/python3.13",
		"include/python3.13/pythread.h",
		"lib/libpython3.13.dylib",
		"lib/python3.13/config-3.13-darwin/Setup.local",
		"lib/python3.13/encodings/__init__.py",
		"lib/python3.13/site.py",
		"lib/python3.13/zipfile/__init__.py",
		"lib/python3.13/zipimport.py",
		"lib/python3.13/site-packages/axiom_encode/__init__.py",
		"lib/python3.13/site-packages/sitecustomize_helper.py",
		"lib/python3.13/site-packages/pkg/data.pth.txt",
		// The stdlib zip name away from lib/ is a plain data file.
		"lib/python3.13/site-packages/pkg/python313.zip",
	} {
		if err := os.WriteFile(filepath.Join(root, path), []byte("\n"), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	return root
}

func plantFixturePath(t *testing.T, root, relative string, directory bool) {
	t.Helper()
	path := filepath.Join(root, filepath.FromSlash(relative))
	if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
		t.Fatal(err)
	}
	if directory {
		if err := os.Mkdir(path, 0o700); err != nil {
			t.Fatal(err)
		}
		path = filepath.Join(path, "__init__.py")
	}
	if err := os.WriteFile(path, []byte("\n"), 0o600); err != nil {
		t.Fatal(err)
	}
}

// The getpath forms run under the supervisor's own -I -S start, before the
// trusted bootstrap; the site forms run in any site-enabled start of the same
// interpreter (the -I launcher shebang, -I children). getpath appends ._pth to
// the full file names of the interpreter and of libpython. The entries marked
// "guard" do not run, because site and the import system compare these names
// case-sensitively; the rule refuses them because it matches names in any
// case. The test plants each directory entry as a directory holding
// __init__.py.
func TestFixtureTrustedTreeRefusesPythonStartupCarriers(t *testing.T) {
	refused := []struct {
		name      string
		path      string
		directory bool
	}{
		// Refused at 4a11230b already; kept as regression guards.
		{"site pth", "lib/python3.13/site-packages/evil.pth", false},
		{"site pth upper case (guard)", "lib/python3.13/site-packages/EVIL.PTH", false},
		{"sitecustomize source", "lib/python3.13/site-packages/sitecustomize.py", false},
		{"pyvenv cfg", "pyvenv.cfg", false},
		{"sitecustomize mixed case (guard)", "lib/python3.13/SiteCustomize.py", false},
		{"editable finder", "lib/python3.13/site-packages/__editable___x_finder.py", false},
		// Accepted at 4a11230b.
		{"sitecustomize package", "lib/python3.13/site-packages/sitecustomize", true},
		{"sitecustomize package in stdlib", "lib/python3.13/sitecustomize", true},
		{"sitecustomize package upper case (guard)", "lib/python3.13/site-packages/SITECUSTOMIZE", true},
		{"usercustomize package", "lib/python3.13/site-packages/usercustomize", true},
		{"sitecustomize sourceless bytecode", "lib/python3.13/site-packages/sitecustomize.pyc", false},
		{"sitecustomize extension", "lib/python3.13/site-packages/sitecustomize.cpython-313-darwin.so", false},
		{"sitecustomize abi3 extension", "lib/python3.13/site-packages/sitecustomize.abi3.so", false},
		{"usercustomize extension", "lib/python3.13/site-packages/usercustomize.so", false},
		{"underscore pth beside interpreter", "bin/python3.13._pth", false},
		{"underscore pth upper case", "bin/PYTHON3.13._PTH", false},
		{"underscore pth beside library", "lib/libpython3.13.dylib._pth", false},
		{"underscore pth beside library upper case", "lib/LIBPYTHON3.13.DYLIB._PTH", false},
		{"pybuilddir beside interpreter", "bin/pybuilddir.txt", false},
		{"pybuilddir upper case", "bin/PYBUILDDIR.TXT", false},
	}
	for _, carrier := range refused {
		t.Run(carrier.name, func(t *testing.T) {
			root := newFixtureRuntimeTree(t)
			if _, err := validateTrustedTree(root); err != nil {
				t.Fatalf("clean fixture tree must pass before planting: %v", err)
			}
			plantFixturePath(t, root, carrier.path, carrier.directory)
			_, err := validateTrustedTree(root)
			if err == nil || !strings.Contains(err.Error(), "forbidden startup") {
				t.Fatalf("expected startup-carrier refusal for %s, got %v", carrier.path, err)
			}
		})
	}
}

func validateFixtureChain(fixture fixturePythonChain) error {
	_, err := validateTrustedExecutionChain(
		fixture.executable,
		[]string{fixture.runtimeRoot},
		[]string{fixture.importRoot},
		fixture.packageRoot,
	)
	return err
}

// sys.path lists <prefix>/lib/python3XY[t].zip first even when it is absent,
// and CPython imports from it as a zip file or as a plain directory, under
// -I -S and before the trusted bootstrap. The same name elsewhere is data.
func TestFixtureTrustedChainRefusesStdlibZipInLibraryDirectory(t *testing.T) {
	for _, carrier := range []struct {
		name      string
		path      string
		directory bool
	}{
		{"stdlib zip", "lib/python313.zip", false},
		{"stdlib zip upper case", "lib/PYTHON313.ZIP", false},
		{"free-threaded stdlib zip", "lib/python313t.zip", false},
		{"stdlib zip as a directory", "lib/python313.zip", true},
		{"stdlib zip under lib64", "lib64/python313.zip", false},
	} {
		t.Run(carrier.name, func(t *testing.T) {
			fixture := newFixturePythonChain(t)
			plantFixturePath(
				t,
				fixture.runtimeRoot,
				"lib/python3.13/site-packages/pkg/python313.zip",
				false,
			)
			if err := validateFixtureChain(fixture); err != nil {
				t.Fatalf("a stdlib zip name outside lib/ must stay accepted: %v", err)
			}
			plantFixturePath(t, fixture.runtimeRoot, carrier.path, carrier.directory)
			err := validateFixtureChain(fixture)
			if err == nil || !strings.Contains(err.Error(), "forbidden startup") {
				t.Fatalf("expected stdlib zip refusal for %s, got %v", carrier.path, err)
			}
		})
	}
}

// On Linux and on some macOS builds, getpath searches for the stdlib
// landmark from the interpreter's own directory upward and treats
// <dir>/Modules/Setup.local as a build tree, under -I -S as well, so a
// directory beside the interpreter can move sys.prefix. A real Setup.local
// sits under lib/python3.X/config-*/, which newFixtureRuntimeTree keeps
// accepted.
func TestFixtureTrustedChainRefusesDirectoryBesideInterpreter(t *testing.T) {
	for _, planted := range []string{
		"bin/Modules/Setup.local",
		"bin/modules/Setup.local",
		"bin/Lib/os.py",
		"bin/lib/python3.13/os.py",
		"bin/lib/python3.13/lib-dynload/.keep",
		"bin/empty/.keep",
	} {
		t.Run(planted, func(t *testing.T) {
			fixture := newFixturePythonChain(t)
			for _, sibling := range []string{"bin/git", "bin/python3.13-config"} {
				plantFixturePath(t, fixture.runtimeRoot, sibling, false)
			}
			if err := validateFixtureChain(fixture); err != nil {
				t.Fatalf("regular files beside the interpreter must stay accepted: %v", err)
			}
			plantFixturePath(t, fixture.runtimeRoot, planted, false)
			err := validateFixtureChain(fixture)
			if err == nil || !strings.Contains(err.Error(), "must hold only regular files") {
				t.Fatalf("expected interpreter-directory refusal for %s, got %v", planted, err)
			}
		})
	}
}

// On Linux and on some macOS builds, getpath's stdlib landmark search starts
// at the interpreter's directory and walks up, so an interpreter deeper in the
// runtime would leave directories between it and the runtime root where a
// lib/python3.X tree could move sys.prefix. The chain keeps the interpreter
// directly in <runtime>/bin, the directory the layout rule limits to regular
// files.
func TestFixtureTrustedChainRequiresInterpreterInRuntimeBin(t *testing.T) {
	for _, relocated := range []string{
		"python3.13",
		"opt/bin/python3.13",
		"libexec/python3.13",
		"bin/sub/python3.13",
	} {
		t.Run(relocated, func(t *testing.T) {
			fixture := newFixturePythonChain(t)
			if err := validateFixtureChain(fixture); err != nil {
				t.Fatalf("an interpreter in <runtime>/bin must be accepted: %v", err)
			}
			interpreter := filepath.Join(fixture.runtimeRoot, filepath.FromSlash(relocated))
			if err := os.MkdirAll(filepath.Dir(interpreter), 0o700); err != nil {
				t.Fatal(err)
			}
			if err := os.WriteFile(interpreter, []byte{0x7f, 'E', 'L', 'F'}, 0o700); err != nil {
				t.Fatal(err)
			}
			launcher := []byte("#!" + interpreter + " -I\n")
			if err := os.WriteFile(fixture.executable, launcher, 0o700); err != nil {
				t.Fatal(err)
			}
			err := validateFixtureChain(fixture)
			if err == nil || !strings.Contains(err.Error(), "directly in the trusted runtime's bin directory") {
				t.Fatalf("expected interpreter-placement refusal for %s, got %v", relocated, err)
			}
		})
	}
}

// macOS builds that derive getpath's interpreter directory from libpython's
// location, such as CPython 3.13.13 and 3.14.4, take the runtime root, so a
// root-level Modules/Setup.local marks a build tree there: sys.prefix moves to
// the build's configured prefix, and that prefix's stdlib zip comes first on
// sys.path. CPython 3.14.0 starts its prefix search in lib/, where
// lib/lib/python3XY.zip moves sys.prefix to lib/.
func TestFixtureTrustedChainRefusesBuildTreeMarkerInRuntimeRoot(t *testing.T) {
	for _, planted := range []string{
		"Modules/Setup.local",
		"MODULES/SETUP.LOCAL",
		"lib/Modules/Setup.local",
		"lib/lib/python313.zip",
		"lib/LIB/python3.13/os.py",
	} {
		t.Run(planted, func(t *testing.T) {
			fixture := newFixturePythonChain(t)
			plantFixturePath(
				t,
				fixture.runtimeRoot,
				"lib/python3.13/config-3.13-darwin/Setup.local",
				false,
			)
			if err := validateFixtureChain(fixture); err != nil {
				t.Fatalf("the real Setup.local under config-*/ must stay accepted: %v", err)
			}
			plantFixturePath(t, fixture.runtimeRoot, planted, false)
			err := validateFixtureChain(fixture)
			if err == nil || !strings.Contains(err.Error(), "forbidden startup") {
				t.Fatalf("expected build-tree marker refusal for %s, got %v", planted, err)
			}
		})
	}
}

// The fixture policy accepts a current-user-owned lib/ above the runtime root
// that only its owner can write, as the production policy accepts a
// root-owned /usr/lib; it still refuses one that others can write.
func TestFixtureTrustedChainChecksLibraryDirectoryAboveRoot(t *testing.T) {
	fixture := newFixturePythonChain(t)
	library := filepath.Join(filepath.Dir(fixture.runtimeRoot), "lib")
	if err := os.Mkdir(library, 0o700); err != nil {
		t.Fatal(err)
	}
	if err := validateFixtureChain(fixture); err != nil {
		t.Fatalf("the chain must accept a trusted library directory above the root: %v", err)
	}
	if err := os.Chmod(library, 0o777); err != nil {
		t.Fatal(err)
	}
	err := validateFixtureChain(fixture)
	if err == nil || !strings.Contains(err.Error(), "untrusted Python library directory above the runtime") {
		t.Fatalf("expected a writable library directory above the root to refuse the runtime, got %v", err)
	}
}
