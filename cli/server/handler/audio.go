// Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package handler

import (
	"encoding/base64"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"os"
	"path"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/openai/openai-go/v3"
	geniex_bridge "github.com/qcom-it-nexa-ai/geniex/bindings/go"
	"github.com/qcom-it-nexa-ai/geniex/internal/types"
	"github.com/qcom-it-nexa-ai/geniex/server/service"
	"github.com/qcom-it-nexa-ai/geniex/server/utils"
)

const speechSSEChunkSize = 4096

func Speech(c *gin.Context) {
	param := openai.AudioSpeechNewParams{}
	if err := c.ShouldBindJSON(&param); err != nil {
		c.JSON(http.StatusBadRequest, map[string]any{"error": err.Error()})
		return
	}
	if param.Speed.Value == 0 {
		param.Speed.Value = 1.0
	}
	slog.Info("Speech request received",
		"model", param.Model,
		"input", param.Input,
		"voice", param.Voice,
		"speed", param.Speed,
		"stream_format", param.StreamFormat,
	)

	audioSpeech, err := service.KeepAliveGet[geniex_bridge.TTS](
		param.Model,
		types.ModelParam{},
		c.GetHeader("GenieX-KeepCache") != "true",
	)
	if err != nil {
		c.JSON(http.StatusInternalServerError, map[string]any{"error": err.Error(), "code": geniex_bridge.SDKErrorCode(err)})
		return
	}

	if param.Input == "" {
		c.JSON(http.StatusOK, nil)
		return
	}

	outputPath := fmt.Sprintf("audio_speech_output_%d.wav", time.Now().UnixNano())
	defer os.Remove(outputPath)
	out, err := audioSpeech.Synthesize(
		geniex_bridge.TtsSynthesizeInput{
			TextUTF8: param.Input,
			Config: &geniex_bridge.TTSConfig{
				Voice: string(param.Voice),
				Speed: float32(param.Speed.Value),
			},
			OutputPath: outputPath,
		})
	if err != nil {
		c.JSON(http.StatusInternalServerError, map[string]any{"error": err.Error(), "code": geniex_bridge.SDKErrorCode(err)})
		return
	}

	if param.StreamFormat == openai.AudioSpeechNewParamsStreamFormatSSE {
		speechStreamSSE(c, outputPath, out.ProfileData)
		return
	}
	c.File(outputPath)
}

func speechStreamSSE(c *gin.Context, audioPath string, profile geniex_bridge.ProfileData) {
	f, err := os.Open(audioPath)
	if err != nil {
		c.JSON(http.StatusInternalServerError, map[string]any{"error": err.Error()})
		return
	}
	defer f.Close()

	c.Header("Content-Type", "text/event-stream")
	c.Header("Cache-Control", "no-cache")
	c.Header("Connection", "keep-alive")

	buf := make([]byte, speechSSEChunkSize)
	for {
		n, err := f.Read(buf)
		if n > 0 {
			chunk := map[string]string{
				"type":  "speech.audio.delta",
				"audio": base64.StdEncoding.EncodeToString(buf[:n]),
			}
			c.SSEvent("", chunk)
			c.Writer.Flush()
		}
		if err == io.EOF {
			break
		}
		if err != nil {
			c.JSON(http.StatusInternalServerError, map[string]any{"error": err.Error()})
			return
		}
	}

	done := map[string]any{
		"type": "speech.audio.done",
		"usage": map[string]int64{
			"input_tokens":  profile.PromptTokens,
			"output_tokens": profile.GeneratedTokens,
			"total_tokens":  profile.TotalTokens(),
		},
	}
	c.SSEvent("", done)
	c.Writer.Flush()
}

func Transcriptions(c *gin.Context) {
	param := openai.AudioTranscriptionNewParams{}
	param.Model = c.PostForm("model")
	stream := c.PostForm("stream")

	if stream == "true" {
		c.JSON(http.StatusBadRequest, map[string]any{"error": "streaming not supported"})
		return
	}

	slog.Info("Transcriptions request received",
		"model", param.Model,
		"stream", stream,
	)

	p, err := service.KeepAliveGet[geniex_bridge.ASR](
		string(param.Model),
		types.ModelParam{},
		false,
	)
	if err != nil {
		c.JSON(http.StatusInternalServerError, map[string]any{"error": err.Error(), "code": geniex_bridge.SDKErrorCode(err)})
		return
	}

	// retrieve file from form data
	file, err := c.FormFile("file")
	if err != nil {
		if err == http.ErrMissingFile {
			// warm up
			c.JSON(http.StatusOK, nil)
			return
		}
		c.JSON(http.StatusBadRequest, map[string]any{"error": "failed to get file: " + err.Error()})
		return
	}
	param.File, err = file.Open()
	if err != nil {
		c.JSON(http.StatusBadRequest, map[string]any{"error": "failed to open file: " + err.Error()})
		return
	}
	data, err := io.ReadAll(param.File)
	if err != nil {
		c.JSON(http.StatusBadRequest, map[string]any{"error": "failed to read file: " + err.Error()})
		return
	}

	// write data to a temp file
	tmpFile, err := os.CreateTemp("", "uri-*"+path.Ext(file.Filename))
	if err != nil {
		c.JSON(http.StatusInternalServerError, map[string]any{"error": "failed to create temp file: " + err.Error()})
		return
	}
	_, err = tmpFile.Write(data)
	if err != nil {
		c.JSON(http.StatusInternalServerError, map[string]any{"error": "failed to write temp file: " + err.Error()})
		return
	}
	tmpFile.Close()
	defer os.Remove(tmpFile.Name())

	res, err := p.Transcribe(geniex_bridge.AsrTranscribeInput{
		AudioPath: tmpFile.Name(),
	})
	result := openai.Transcription{
		Text: res.Result.Transcript,
	}
	if err != nil {
		c.JSON(http.StatusInternalServerError, map[string]any{"error": err.Error(), "code": geniex_bridge.SDKErrorCode(err)})
	} else {
		c.JSON(http.StatusOK, result)
	}
}

type DiarizeRequest struct {
	Model string `json:"model" binding:"required"`
	Audio string `json:"audio"`
}

func Diarize(c *gin.Context) {
	param := DiarizeRequest{}
	if err := c.ShouldBindJSON(&param); err != nil {
		c.JSON(http.StatusBadRequest, map[string]any{"error": err.Error()})
		return
	}

	slog.Info("Diarize request received",
		"model", param.Model,
		"audio", param.Audio,
	)

	p, err := service.KeepAliveGet[geniex_bridge.Diarize](
		string(param.Model),
		types.ModelParam{},
		false,
	)
	if err != nil {
		c.JSON(http.StatusInternalServerError, map[string]any{"error": err.Error(), "code": geniex_bridge.SDKErrorCode(err)})
		return
	}

	// warm up
	if param.Audio == "" {
		c.JSON(http.StatusOK, nil)
		return
	}

	file, err := utils.SaveURIToTempFile(param.Audio)
	if err != nil {
		c.JSON(http.StatusBadRequest, map[string]any{"error": "failed to save audio: " + err.Error()})
		return
	}
	defer os.Remove(file)
	res, err := p.Infer(geniex_bridge.DiarizeInferInput{
		AudioPath: file,
	})
	if err != nil {
		c.JSON(http.StatusInternalServerError, map[string]any{"error": err.Error(), "code": geniex_bridge.SDKErrorCode(err)})
	} else {
		c.JSON(http.StatusOK, res)
	}
}
