//go:build !darwin && !linux

package main

import (
	"fmt"
	"os"
)

func main() {
	fmt.Fprintln(os.Stderr, "axiom-encode-apply-signer supports only Linux and macOS")
	os.Exit(2)
}
