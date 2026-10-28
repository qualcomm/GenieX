// SPDX-License-Identifier: BSD-3-Clause
package geniex_sdk

import (
	"os"
	"testing"
)

func TestTTSContracts(t *testing.T) {
	if os.Getenv("GENIEX_TTS_CONTRACT_TEST") != "1" {
		t.Skip("native TTS test plugins required")
	}
	if err := Init(); err != nil {
		t.Fatal(err)
	}
	defer DeInit()
	model, err := NewTTS(TtsCreateInput{PluginID: "mock_tts", ModelPath: "unused", Device: "npu"})
	if err != nil {
		t.Fatal(err)
	}
	defer model.Close()
	frames := 0
	result, err := model.Synthesize("test", "", func(pcm []float32, rate uint32) bool {
		frames++
		if rate != 24000 || len(pcm) != 4 || pcm[1] != 0.5 {
			t.Error("PCM mismatch")
		}
		return true
	})
	if err != nil || result.Samples != 40 || frames != 10 {
		t.Fatalf("stream: %+v %v", result, err)
	}
	result, err = model.Synthesize("test", "", func([]float32, uint32) bool { return false })
	if err != nil || !result.Cancelled || result.Samples != 4 {
		t.Fatalf("cancel: %+v %v", result, err)
	}
	_, err = model.Synthesize("test", "", func([]float32, uint32) bool { panic("callback failure") })
	if err == nil {
		t.Fatal("callback panic not reported")
	}
	result, err = model.Synthesize("test", "", func([]float32, uint32) bool { return true })
	if err != nil || result.Cancelled {
		t.Fatalf("reuse: %+v %v", result, err)
	}
	if err := model.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err := model.Synthesize("test", "", func([]float32, uint32) bool { return true }); err == nil {
		t.Fatal("closed handle accepted")
	}
}

func TestTTSPhononNPU(t *testing.T) {
	bundle := os.Getenv("GENIEX_PHONON_TEST_BUNDLE")
	if bundle == "" {
		t.Skip("compiled Phonon bundle required")
	}
	if err := SetQairtRuntimePath(os.Getenv("GENIEX_PHONON_TEST_RUNTIME")); err != nil {
		t.Fatal(err)
	}
	if err := Init(); err != nil {
		t.Fatal(err)
	}
	defer DeInit()
	model, err := NewTTS(TtsCreateInput{PluginID: "phonon_qnn", ModelPath: bundle, Device: "npu",
		RuntimePath: os.Getenv("GENIEX_PHONON_TEST_RUNTIME"), TextLibrary: os.Getenv("GENIEX_PHONON_TEST_TEXT_LIBRARY"),
		Language: "en", SocModel: os.Getenv("GENIEX_PHONON_TEST_SOC"), VerifyNPU: true})
	if err != nil {
		t.Fatal(err)
	}
	defer model.Close()
	for run := 0; run < 2; run++ {
		samples := 0
		result, err := model.Synthesize("Phonon is speaking through the GenieX Go binding.", "",
			func(pcm []float32, rate uint32) bool { samples += len(pcm); return rate == 24000 })
		if err != nil || result.Cancelled || result.Samples != uint64(samples) || samples == 0 ||
			result.PrefillNPUMs <= 0 || result.StepNPUMs <= 0 || result.PrefillCalls != 1 {
			t.Fatalf("NPU run %d: %+v %v", run, result, err)
		}
		t.Logf("NPU run %d: %+v", run, result)
	}
}
