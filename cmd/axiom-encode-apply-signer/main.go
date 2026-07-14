//go:build darwin || linux

// axiom-encode-apply-signer is the production external Ed25519 signer for the
// Axiom Encode trusted-signing boundary, plus the launcher that wires it to the
// compiled supervisor inside GitHub Actions.
//
// The `serve` subcommand is the leaf signer: it holds one operation-scoped
// private key, ingested from a pipe/socket descriptor (never env, argv, or
// disk), and answers external-signer protocol v2 over a pre-connected Unix
// socket. It signs only its provisioned scope and only the two domain-bound
// message shapes protocol v2 defines. It refuses to run outside GitHub Actions
// unless --allow-local-dev, which self-generates a throwaway key.
//
// The `run` subcommand is the process manager: it reads the key from a named
// environment variable and clears it before spawning any child, pre-opens the
// connected signer socket, streams the key to the signer over a pipe, and
// executes the trusted supervisor with the signer attached on --apply-signer-fd.
package main

import (
	"flag"
	"fmt"
	"io"
	"os"
)

const buildKind = "production"

type multiFlag []string

func (values *multiFlag) String() string { return fmt.Sprintf("%v", []string(*values)) }

func (values *multiFlag) Set(value string) error {
	*values = append(*values, value)
	return nil
}

func main() {
	if len(os.Args) == 2 && os.Args[1] == "--build-kind" {
		fmt.Println(buildKind)
		return
	}
	if len(os.Args) < 2 {
		fmt.Fprintln(os.Stderr, "usage: axiom-encode-apply-signer <serve|run> [flags]")
		os.Exit(2)
	}
	if os.Args[1] == "serve" || os.Args[1] == "run" {
		// Harden before ANY argument parsing or key handling. execve reset
		// PR_SET_DUMPABLE to 1; on the launcher path the signing key is already
		// resident in this process's inherited environment. Denying core dumps,
		// ptrace, and /proc/self/{environ,fd} access as the first instruction
		// closes that window before flags are read.
		if err := hardenProcess(); err != nil {
			fmt.Fprintf(os.Stderr, "apply signer: could not harden process: %v\n", err)
			os.Exit(2)
		}
	}
	switch os.Args[1] {
	case "serve":
		options, err := parseServeOptions(os.Args[2:])
		if err != nil {
			fmt.Fprintf(os.Stderr, "apply signer: %v\n", err)
			os.Exit(2)
		}
		if err := runServe(options, os.Getenv, os.Stdout); err != nil {
			fmt.Fprintf(os.Stderr, "apply signer: %v\n", err)
			os.Exit(2)
		}
	case "run":
		options, err := parseRunOptions(os.Args[2:])
		if err != nil {
			fmt.Fprintf(os.Stderr, "apply signer launcher: %v\n", err)
			os.Exit(2)
		}
		exitCode, err := runLauncher(options, os.Getenv)
		if err != nil {
			fmt.Fprintf(os.Stderr, "apply signer launcher: %v\n", err)
			os.Exit(2)
		}
		os.Exit(exitCode)
	default:
		fmt.Fprintf(os.Stderr, "apply signer: unknown subcommand %q\n", os.Args[1])
		os.Exit(2)
	}
}

func parseServeOptions(arguments []string) (serveOptions, error) {
	flags := flag.NewFlagSet("serve", flag.ContinueOnError)
	flags.SetOutput(io.Discard)
	scope := flags.String("scope", "", "operation scope this signer serves (apply_ed25519 or eval_ed25519)")
	socketFD := flags.Int("socket-fd", -1, "inherited connected Unix stream socket descriptor")
	keyFD := flags.Int("key-fd", -1, "inherited pipe/socket descriptor carrying the private key")
	readyFD := flags.Int("ready-fd", -1, "optional descriptor to signal after hardening, before reading the key")
	repository := flags.String("expected-github-repository", "", "required GITHUB_REPOSITORY value")
	allowLocalDev := flags.Bool("allow-local-dev", false, "run outside GitHub Actions with a self-generated throwaway key")
	var refs multiFlag
	var events multiFlag
	flags.Var(&refs, "allowed-workflow-ref", "allowed GITHUB_WORKFLOW_REF value; repeatable")
	flags.Var(&events, "allowed-event-name", "allowed GITHUB_EVENT_NAME value; repeatable")
	if err := flags.Parse(arguments); err != nil {
		return serveOptions{}, err
	}
	if len(flags.Args()) != 0 {
		return serveOptions{}, fmt.Errorf("unexpected arguments: %v", flags.Args())
	}
	if *socketFD < 0 {
		return serveOptions{}, fmt.Errorf("--socket-fd is required")
	}
	return serveOptions{
		scope:    *scope,
		socketFD: *socketFD,
		keyFD:    *keyFD,
		readyFD:  *readyFD,
		binding: contextBinding{
			expectedRepository:  *repository,
			allowedWorkflowRefs: refs,
			allowedEventNames:   events,
			allowLocalDev:       *allowLocalDev,
		},
	}, nil
}

func parseRunOptions(arguments []string) (runOptions, error) {
	flags := flag.NewFlagSet("run", flag.ContinueOnError)
	flags.SetOutput(io.Discard)
	scope := flags.String("scope", "", "operation scope to provision (apply_ed25519 or eval_ed25519)")
	keyEnv := flags.String("key-env", "", "name of the environment variable holding the base64/PEM private key")
	supervisor := flags.String("supervisor", "", "path to the compiled axiom-encode-signing-supervisor")
	trustRoots := flags.String("trusted-signing-roots", "", "protected three-root trust config for the supervisor")
	packageRoot := flags.String("trusted-python-package-root", "", "supervisor --trusted-python-package-root")
	repository := flags.String("expected-github-repository", "", "required GITHUB_REPOSITORY value")
	var runtimeRoots multiFlag
	var importRoots multiFlag
	var refs multiFlag
	var events multiFlag
	flags.Var(&runtimeRoots, "trusted-python-runtime-root", "supervisor --trusted-python-runtime-root; repeatable")
	flags.Var(&importRoots, "trusted-python-import-root", "supervisor --trusted-python-import-root; repeatable")
	flags.Var(&refs, "allowed-workflow-ref", "allowed GITHUB_WORKFLOW_REF value; repeatable")
	flags.Var(&events, "allowed-event-name", "allowed GITHUB_EVENT_NAME value; repeatable")
	if err := flags.Parse(arguments); err != nil {
		return runOptions{}, err
	}
	command := flags.Args()
	if len(command) == 0 {
		return runOptions{}, fmt.Errorf("expected `-- axiom-encode encode ... --apply` after launcher flags")
	}
	if *supervisor == "" {
		return runOptions{}, fmt.Errorf("--supervisor is required")
	}
	if *trustRoots == "" {
		return runOptions{}, fmt.Errorf("--trusted-signing-roots is required")
	}
	return runOptions{
		scope:              *scope,
		keyEnv:             *keyEnv,
		supervisor:         *supervisor,
		trustRoots:         *trustRoots,
		pythonRuntimeRoots: runtimeRoots,
		pythonImportRoots:  importRoots,
		pythonPackageRoot:  *packageRoot,
		binding: contextBinding{
			expectedRepository:  *repository,
			allowedWorkflowRefs: refs,
			allowedEventNames:   events,
			// The launcher is CI-only; local-dev bypass lives on the leaf signer.
			allowLocalDev: false,
		},
		command: command,
	}, nil
}
