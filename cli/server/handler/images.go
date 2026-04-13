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
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/openai/openai-go/v3"

	geniex_bridge "github.com/qcom-it-nexa-ai/geniex/bindings/go"
	"github.com/qcom-it-nexa-ai/geniex/internal/types"
	"github.com/qcom-it-nexa-ai/geniex/server/service"
)

func ImageGenerations(c *gin.Context) {
	param := openai.ImageGenerateParams{}
	if err := c.ShouldBindJSON(&param); err != nil {
		slog.Error("Failed to bind JSON request", "error", err)
		c.JSON(http.StatusBadRequest, map[string]any{"error": err.Error()})
		return
	}

	slog.Info("Image generation request received",
		"model", param.Model,
		"prompt_length", len(param.Prompt),
		"n", param.N,
		"size", param.Size)

	if param.N.Value == 0 {
		param.N.Value = 1
	}
	if param.Size == "" {
		param.Size = openai.ImageGenerateParamsSize256x256
	}
	if param.ResponseFormat != "" && param.ResponseFormat != openai.ImageGenerateParamsResponseFormatB64JSON {
		c.JSON(http.StatusBadRequest, map[string]any{"error": "only 'b64_json' response format is supported"})
		return
	}

	imageGen, err := service.KeepAliveGet[geniex_bridge.ImageGen](
		param.Model,
		types.ModelParam{},
		c.GetHeader("GenieX-KeepCache") != "true",
	)
	if err != nil {
		c.JSON(http.StatusInternalServerError, map[string]any{"error": err.Error(), "code": geniex_bridge.SDKErrorCode(err)})
		return
	}

	// warm up
	if param.Prompt == "" {
		c.JSON(http.StatusOK, nil)
		return
	}

	width, height, err := parseImageSize(string(param.Size))
	if err != nil {
		c.JSON(http.StatusBadRequest, map[string]any{"error": err.Error()})
		return
	}

	var imageData []openai.Image
	slog.Info("Starting image generation", "count", param.N.Value, "size", string(param.Size))
	for i := range param.N.Value {
		outputPath := fmt.Sprintf("imagegen_output_%d.png", time.Now().UnixNano())
		defer os.Remove(outputPath)
		slog.Debug("Generating image", "index", i, "output_path", outputPath)

		config := &geniex_bridge.ImageGenerationConfig{
			Prompts:         []string{param.Prompt},
			NegativePrompts: []string{"blurry, low quality, distorted, low resolution"},
			Height:          height,
			Width:           width,
			SamplerConfig: geniex_bridge.ImageSamplerConfig{
				Method:        "ddim",
				Steps:         20,
				GuidanceScale: 7.5,
				Eta:           0.0,
				Seed:          int32(time.Now().UnixNano() % 1000000),
			},
			SchedulerConfig: geniex_bridge.SchedulerConfig{
				Type:              "ddim",
				NumTrainTimesteps: 1000,
				StepsOffset:       1,
				BetaStart:         0.00085,
				BetaEnd:           0.012,
				BetaSchedule:      "scaled_linear",
				PredictionType:    "epsilon",
				TimestepType:      "discrete",
				TimestepSpacing:   "leading",
				InterpolationType: "linear",
				ConfigPath:        "",
			},
			Strength: 1.0,
		}

		result, err := imageGen.Txt2Img(geniex_bridge.ImageGenTxt2ImgInput{
			PromptUTF8: param.Prompt,
			Config:     config,
			OutputPath: outputPath,
		})
		if err != nil {
			c.JSON(http.StatusInternalServerError, map[string]any{"error": fmt.Sprintf("image generation failed: %v", err), "code": geniex_bridge.SDKErrorCode(err)})
			return
		}

		data := openai.Image{
			RevisedPrompt: param.Prompt,
		}

		b64Data, err := encodeImageToBase64(result.OutputImagePath)
		if err != nil {
			c.JSON(http.StatusInternalServerError, map[string]any{"error": fmt.Sprintf("failed to encode image: %v", err)})
			return
		}
		data.B64JSON = b64Data

		imageData = append(imageData, data)
		slog.Info("Image generated successfully", "index", i, "output_path", result.OutputImagePath)
	}

	response := openai.ImagesResponse{
		Created: time.Now().Unix(),
		Data:    imageData,
	}

	slog.Info("Image generation completed successfully", "total_images", len(imageData))
	c.JSON(http.StatusOK, response)
}

func parseImageSize(size string) (int32, int32, error) {
	parts := strings.Split(size, "x")
	if len(parts) != 2 {
		return 0, 0, errors.New("invalid size format")
	}

	width, err := strconv.Atoi(parts[0])
	if err != nil {
		return 0, 0, errors.New("invalid width")
	}

	height, err := strconv.Atoi(parts[1])
	if err != nil {
		return 0, 0, errors.New("invalid height")
	}

	return int32(width), int32(height), nil
}

func encodeImageToBase64(imagePath string) (string, error) {
	imageData, err := os.ReadFile(imagePath)
	if err != nil {
		return "", fmt.Errorf("failed to read image file: %v", err)
	}
	mimeType := http.DetectContentType(imageData)
	base64String := base64.StdEncoding.EncodeToString(imageData)
	return fmt.Sprintf("data:%s;base64,%s", mimeType, base64String), nil
}
