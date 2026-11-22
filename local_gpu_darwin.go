//go:build darwin && cgo

package main

import (
	"github.com/gomlx/compute"
	"github.com/gomlx/compute/gobackend"
	"github.com/gomlx/go-darwinml/compute/mps"
)

// useGPU runs the embedding model on the Apple GPU (Metal Performance
// Shaders, part of macOS: nothing to install). On the tycoon set it answered
// a query in 2.9 ms against 8.2 ms on the CPU, with the same results.
// hugot always asks GoMLX for its "go" backend, so MPS is registered under
// that name; where Metal is unavailable, the CPU backend stands in.
// It reports whether the GPU will be used.
func useGPU() bool {
	probe, err := mps.New("")
	if err != nil {
		return false
	}
	probe.Finalize()
	compute.Register("go", func(config string) (compute.Backend, error) {
		if b, err := mps.New(config); err == nil {
			return b, nil
		}
		return gobackend.New(config)
	})
	return true
}
