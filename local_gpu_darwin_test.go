//go:build darwin && cgo

package main

import (
	"context"
	"testing"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/gobackend"
	"github.com/knights-analytics/hugot"
	"github.com/knights-analytics/hugot/options"
	"github.com/knights-analytics/hugot/pipelines"
)

// Vectors made on the GPU (padded, as jrnl serve runs it) and on the CPU
// carry the same embedder ID and are compared with each other, so they
// must agree.
func TestGPUAndCPUVectorsAgree(t *testing.T) {
	needLocalModel(t)
	texts := []string{"Users could not sign in after the token refresh.",
		localQueryPrefix + "why do logins fail after refreshing credentials", "Chose Go over Bun for the CLI."}
	pipeline := func(name string, opts ...options.WithOption) *pipelines.FeatureExtractionPipeline {
		s, err := hugot.NewGoSession(context.Background(), opts...)
		if err != nil {
			t.Fatal(err)
		}
		p, err := hugot.NewPipeline(s, hugot.FeatureExtractionConfig{ModelPath: localDir(), Name: name, OnnxFilename: "model.onnx"})
		if err != nil {
			t.Fatal(err)
		}
		return p
	}
	t.Cleanup(func() { compute.Register("go", gobackend.New) })
	compute.Register("go", gobackend.New)
	cpu := pipeline("cpu")
	if !useGPU() {
		t.Skip("no GPU")
	}
	gpu := pipeline("gpu", options.WithGoMLXSequenceBuckets(localSeqBuckets), options.WithGoMLXBatchBuckets(localBatchBucket))
	if name := gpu.Model.GoMLXModel.Backend.Name(); name != "mps" {
		t.Fatalf("the GPU pipeline runs on %q, want mps", name)
	}
	if name := cpu.Model.GoMLXModel.Backend.Name(); name == "mps" {
		t.Fatal("the CPU pipeline runs on the GPU")
	}
	rc, err := cpu.RunRaw(context.Background(), texts)
	if err != nil {
		t.Fatal(err)
	}
	rg, err := gpu.RunRaw(context.Background(), texts)
	if err != nil {
		t.Fatal(err)
	}
	for i := range texts {
		if c := dot(normalize(rc.HiddenStates[i][0]), normalize(rg.HiddenStates[i][0])); c < 0.999 {
			t.Errorf("text %d: GPU and CPU vectors differ, cosine %.5f", i, c)
		}
	}
}
