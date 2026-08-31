//go:build !linux

package main

import (
	"fmt"
	"os"
)

func main() {
	if len(os.Args) == 2 && os.Args[1] == "--build-kind" {
		fmt.Println("unsupported")
		return
	}
	fmt.Fprintln(os.Stderr, "axiom-encode-process-guardian is supported only on Linux")
	os.Exit(2)
}
