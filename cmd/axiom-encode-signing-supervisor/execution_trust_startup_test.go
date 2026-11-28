//go:build darwin || linux

package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestPythonStartupCarrierNames(t *testing.T) {
	for name, want := range map[string]bool{
		// getpath forms (honored under -I -S, before the bootstrap)
		"python3.13._pth": true,
		"PYTHON3.13._PTH": true,
		"pybuilddir.txt":  true,
		"PyBuildDir.txt":  true,
		"pyvenv.cfg":      true,
		"PYVENV.CFG":      true,
		// site forms (honored by any site-enabled start)
		"evil.pth":                            true,
		"x.egg-link":                          true,
		"__editable___x_finder.py":            true,
		"sitecustomize":                       true,
		"sitecustomize.py":                    true,
		"sitecustomize.pyc":                   true,
		"sitecustomize.cpython-313-darwin.so": true,
		"sitecustomize.abi3.so":               true,
		"usercustomize":                       true,
		"usercustomize.cpython-313-x86_64-linux-gnu.so": true,
		// case variants that site and the import system do not honor; the
		// rule refuses them anyway because it matches names in any case
		"EVIL.PTH":         true,
		"SiteCustomize.py": true,
		"SITECUSTOMIZE":    true,
		// ordinary runtime names
		"python3.13":              false,
		"python3":                 false,
		"libpython3.13.dylib":     false,
		"Setup.local":             false,
		"site.py":                 false,
		"zipfile":                 false,
		"zipimport.py":            false,
		"sitecustomize_helper.py": false,
		"data.pth.txt":            false,
		"pythread.h":              false,
		"encodings":               false,
		"Modules":                 false,
		"lib":                     false,
		// the stdlib zip is positional (validateTrustedRuntimeLayout)
		"python313.zip": false,
	} {
		if got := isPythonStartupCarrierName(name); got != want {
			t.Errorf("isPythonStartupCarrierName(%q) = %t, want %t", name, got, want)
		}
	}
}

func TestStdlibZipNames(t *testing.T) {
	for name, want := range map[string]bool{
		"python313.zip":  true,
		"PYTHON313.ZIP":  true,
		"python313t.zip": true,
		"python314.zip":  true,
		"python3.zip":    true,
		"python313.ziq":  false,
		"python2.7.zip":  false,
		"libpython.zip":  false,
		"python313":      false,
		"zipimport.py":   false,
	} {
		if got := isStdlibZipName(name); got != want {
			t.Errorf("isStdlibZipName(%q) = %t, want %t", name, got, want)
		}
	}
}

// newLayoutRuntime builds <temp>/runtime with the interpreter at the given
// slash-separated path and a stdlib under lib/. validateTrustedRuntimeLayout
// checks owners and modes only for a lib/ or lib64/ above the runtime root,
// and the tree has none of its own, so these tests run under the production
// and fixture trust policies alike.
func newLayoutRuntime(t *testing.T, interpreterPath string) (string, string) {
	t.Helper()
	temporary, err := filepath.EvalSymlinks(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	root := filepath.Join(temporary, "runtime")
	interpreter := filepath.Join(root, filepath.FromSlash(interpreterPath))
	for _, path := range []string{
		interpreter,
		filepath.Join(filepath.Dir(interpreter), "python3.13-config"),
		filepath.Join(root, "lib", "libpython3.13.dylib"),
		filepath.Join(root, "lib", "python3.13", "os.py"),
		filepath.Join(root, "lib", "python3.13", "config-3.13-darwin", "Setup.local"),
		filepath.Join(root, "lib", "python3.13", "site-packages", "pkg", "python313.zip"),
	} {
		if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, []byte("\n"), 0o700); err != nil {
			t.Fatal(err)
		}
	}
	return root, interpreter
}

func plantLayoutPath(t *testing.T, root, relative string, directory bool) {
	t.Helper()
	path := filepath.Join(root, filepath.FromSlash(relative))
	if directory {
		if err := os.MkdirAll(path, 0o700); err != nil {
			t.Fatal(err)
		}
		return
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte("\n"), 0o600); err != nil {
		t.Fatal(err)
	}
}

func TestTrustedRuntimeLayoutRefusesNonRegularEntryBesideInterpreter(t *testing.T) {
	for _, planted := range []string{
		"bin/Modules",
		"bin/Lib",
		"bin/lib",
		"bin/__pycache__",
	} {
		t.Run(planted, func(t *testing.T) {
			root, interpreter := newLayoutRuntime(t, "bin/python3.13")
			if err := validateTrustedRuntimeLayout(root, interpreter); err != nil {
				t.Fatalf("clean layout must pass before planting: %v", err)
			}
			plantLayoutPath(t, root, planted, true)
			err := validateTrustedRuntimeLayout(root, interpreter)
			if err == nil || !strings.Contains(err.Error(), "must hold only regular files") {
				t.Fatalf("expected interpreter-directory refusal for %s, got %v", planted, err)
			}
		})
	}
}

func TestTrustedRuntimeLayoutRefusesStdlibZipSlot(t *testing.T) {
	for _, carrier := range []struct {
		interpreter string
		path        string
		directory   bool
	}{
		{"bin/python3.13", "lib/python313.zip", false},
		{"bin/python3.13", "lib/PYTHON313.ZIP", false},
		{"bin/python3.13", "lib/python313t.zip", false},
		{"bin/python3.13", "lib/python313.zip", true},
		{"bin/python3.13", "lib64/python313.zip", false},
		// getpath's search for the stdlib zip runs up from the interpreter's
		// directory and does not stop at the runtime root, so lib/ of every
		// ancestor up to / counts. Paths starting with ../ land beside the
		// runtime root, still inside the test's temporary directory.
		{"opt/bin/python3.13", "opt/lib/python313.zip", false},
		{"opt/bin/python3.13", "lib/python313.zip", false},
		{"bin/python3.13", "../lib/python313.zip", false},
		{"opt/bin/python3.13", "../lib64/PYTHON313.ZIP", false},
	} {
		t.Run(carrier.interpreter+" "+carrier.path, func(t *testing.T) {
			root, interpreter := newLayoutRuntime(t, carrier.interpreter)
			if err := validateTrustedRuntimeLayout(root, interpreter); err != nil {
				t.Fatalf("clean layout must pass before planting: %v", err)
			}
			plantLayoutPath(t, root, carrier.path, carrier.directory)
			err := validateTrustedRuntimeLayout(root, interpreter)
			if err == nil || !strings.Contains(err.Error(), "forbidden startup") {
				t.Fatalf("expected stdlib zip refusal for %s, got %v", carrier.path, err)
			}
		})
	}
}

// getpath treats <directory>/Modules/Setup.local as a build tree in the
// directory it takes as the interpreter's: bin/ on Linux and on some macOS
// builds, the runtime root on macOS builds that derive it from libpython's
// location. CPython 3.14.0 starts its prefix search in lib/, so a lib or
// lib64 entry there moves sys.prefix. The real Setup.local under
// lib/python3.X/config-*/ stays accepted.
func TestTrustedRuntimeLayoutRefusesBuildTreeAndNestedLibraryMarkers(t *testing.T) {
	for _, planted := range []struct {
		path      string
		directory bool
	}{
		{"Modules/Setup.local", false},
		{"MODULES/SETUP.LOCAL", false},
		{"modules", false},
		{"Modules", true},
		{"bin/Modules", false},
		{"bin/MODULES", false},
		{"lib/Modules/Setup.local", false},
		{"lib64/modules", true},
		{"lib/lib/python313.zip", false},
		{"lib/LIB64/python3.13/os.py", false},
		{"lib64/lib", true},
	} {
		t.Run(planted.path, func(t *testing.T) {
			root, interpreter := newLayoutRuntime(t, "bin/python3.13")
			if err := validateTrustedRuntimeLayout(root, interpreter); err != nil {
				t.Fatalf("clean layout must pass before planting: %v", err)
			}
			plantLayoutPath(t, root, planted.path, planted.directory)
			err := validateTrustedRuntimeLayout(root, interpreter)
			if err == nil || !strings.Contains(err.Error(), "forbidden startup") {
				t.Fatalf("expected build-tree marker refusal for %s, got %v", planted.path, err)
			}
		})
	}
}

func TestTrustedRuntimeLayoutRefusesSymlinkBesideInterpreter(t *testing.T) {
	for _, target := range []string{"python3.13", "../lib"} {
		t.Run(target, func(t *testing.T) {
			root, interpreter := newLayoutRuntime(t, "bin/python3.13")
			if err := os.Symlink(target, filepath.Join(root, "bin", "link")); err != nil {
				t.Fatal(err)
			}
			err := validateTrustedRuntimeLayout(root, interpreter)
			if err == nil || !strings.Contains(err.Error(), "must hold only regular files") {
				t.Fatalf("expected interpreter-directory refusal for a symlink to %s, got %v", target, err)
			}
		})
	}
}

// getpath needs only search permission on lib/ to open lib/python3XY.zip, so
// a lib/ the supervisor cannot list could hide one.
func TestTrustedRuntimeLayoutRefusesUnreadableLibraryAboveRoot(t *testing.T) {
	if os.Geteuid() == 0 {
		t.Skip("root reads a directory whatever its mode")
	}
	root, interpreter := newLayoutRuntime(t, "bin/python3.13")
	library := filepath.Join(filepath.Dir(root), "lib")
	if err := os.Mkdir(library, 0o700); err != nil {
		t.Fatal(err)
	}
	if err := os.Chmod(library, 0o311); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = os.Chmod(library, 0o700) })
	err := validateTrustedRuntimeLayout(root, interpreter)
	if err == nil || !strings.Contains(err.Error(), "could not inspect trusted Python library directory") {
		t.Fatalf("expected an unreadable library directory to refuse the runtime, got %v", err)
	}
}

// A lib/ above the runtime root that the caller can write, or that resolves
// into a directory the caller can write, could gain python3XY.zip after
// validation, and getpath would then take its parent as sys.prefix. The
// production policy also refuses these directories because the current
// user, not root, owns them.
func TestTrustedRuntimeLayoutRefusesUntrustedLibraryAboveRoot(t *testing.T) {
	for _, layout := range []struct {
		name     string
		target   string
		writable string
	}{
		{"writable directory", "", "lib"},
		{"symlink to a writable directory", "elsewhere", "elsewhere"},
		{"symlink into a writable directory", "shared/lib", "shared"},
	} {
		t.Run(layout.name, func(t *testing.T) {
			root, interpreter := newLayoutRuntime(t, "bin/python3.13")
			parent := filepath.Dir(root)
			library := filepath.Join(parent, "lib")
			if layout.target != "" {
				library = filepath.Join(parent, layout.target)
				if err := os.Symlink(layout.target, filepath.Join(parent, "lib")); err != nil {
					t.Fatal(err)
				}
			}
			if err := os.MkdirAll(library, 0o700); err != nil {
				t.Fatal(err)
			}
			if err := os.Chmod(filepath.Join(parent, layout.writable), 0o777); err != nil {
				t.Fatal(err)
			}
			err := validateTrustedRuntimeLayout(root, interpreter)
			if err == nil || !strings.Contains(err.Error(), "untrusted Python library directory above the runtime") {
				t.Fatalf("expected a writable library directory above the root to refuse the runtime, got %v", err)
			}
		})
	}
}

func TestTrustedRuntimeLayoutRefusesUnreadableRuntimeRoot(t *testing.T) {
	if os.Geteuid() == 0 {
		t.Skip("root reads any directory")
	}
	root, interpreter := newLayoutRuntime(t, "bin/python3.13")
	if err := os.Chmod(root, 0o311); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = os.Chmod(root, 0o700) })
	err := validateTrustedRuntimeLayout(root, interpreter)
	if err == nil || !strings.Contains(err.Error(), "could not inspect trusted Python runtime directory") {
		t.Fatalf("expected an unreadable runtime root to refuse the runtime, got %v", err)
	}
}

func TestTrustedRuntimeLayoutRefusesDanglingLibrarySymlinkAboveRoot(t *testing.T) {
	root, interpreter := newLayoutRuntime(t, "bin/python3.13")
	if err := os.Symlink("missing", filepath.Join(filepath.Dir(root), "lib64")); err != nil {
		t.Fatal(err)
	}
	err := validateTrustedRuntimeLayout(root, interpreter)
	if err == nil || !strings.Contains(err.Error(), "must resolve to a directory") {
		t.Fatalf("expected a dangling library symlink above the root to refuse the runtime, got %v", err)
	}
}
