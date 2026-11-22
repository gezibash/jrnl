//go:build !darwin || !cgo

package main

// useGPU: elsewhere the model runs on the CPU, in pure Go.
func useGPU() bool { return false }
