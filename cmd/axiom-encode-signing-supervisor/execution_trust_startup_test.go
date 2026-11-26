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
// reads names and entry types only, so the tree needs no particular owner and
// this test runs under the production and fixture trust policies alike.
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
		// An interpreter deeper in the runtime: getpath's upward search can
		// settle on any ancestor of its directory, so each one's lib/ counts.
		{"opt/bin/python3.13", "opt/lib/python313.zip", false},
		{"opt/bin/python3.13", "lib/python313.zip", false},
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
