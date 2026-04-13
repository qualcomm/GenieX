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

package logic

import (
	"fmt"
	"image"
	"image/color"
	"image/draw"
	_ "image/jpeg"
	"image/png"
	"log/slog"
	"os"
	"path/filepath"
	"reflect"
	"runtime"
	"strings"
	"sync"

	"golang.org/x/image/font"
	"golang.org/x/image/font/basicfont"
	"golang.org/x/image/math/fixed"

	geniex_sdk "github.com/qcom-it-nexa-ai/geniex/bindings/go"
)

func drawBBoxes(img *image.RGBA, results []geniex_sdk.CVResult) {
	slog.Debug("Drawing bounding boxes on image", "num_results", len(results))

	bounds := img.Bounds()
	const bboxLineWidth = 2

	d := &font.Drawer{
		Dst:  img,
		Src:  image.NewUniform(color.White),
		Face: basicfont.Face7x13,
	}
	colors := []color.RGBA{
		{R: 255, G: 0, B: 0, A: 255},   // Red
		{R: 0, G: 255, B: 255, A: 255}, // Cyan
		{R: 0, G: 255, B: 0, A: 255},   // Green
		{R: 0, G: 0, B: 255, A: 255},   // Blue
		{R: 255, G: 255, B: 0, A: 255}, // Yellow
		{R: 255, G: 0, B: 255, A: 255}, // Magenta
		{R: 255, G: 165, B: 0, A: 255}, // Orange
		{R: 128, G: 0, B: 128, A: 255}, // Purple
	}
	for _, r := range results {
		if r.BBox.Width > 0 && r.BBox.Height > 0 {
			bboxColor := colors[r.ClassID%int32(len(colors))]
			x, y, w, h := int(r.BBox.X), int(r.BBox.Y), int(r.BBox.Width), int(r.BBox.Height)
			if x < 0 {
				x, w = 0, w+x
			}
			if y < 0 {
				y, h = 0, h+y
			}
			if x+w > bounds.Dx() {
				w = bounds.Dx() - x
			}
			if y+h > bounds.Dy() {
				h = bounds.Dy() - y
			}

			for i := range bboxLineWidth {
				for j := x; j < x+w && j < bounds.Dx(); j++ {
					if y+i < bounds.Dy() {
						img.Set(j, y+i, bboxColor)
					}
					if y+h-1-i >= 0 {
						img.Set(j, y+h-1-i, bboxColor)
					}
				}
				for j := y; j < y+h && j < bounds.Dy(); j++ {
					if x+i < bounds.Dx() {
						img.Set(x+i, j, bboxColor)
					}
					if x+w-1-i >= 0 {
						img.Set(x+w-1-i, j, bboxColor)
					}
				}
			}

			textLabel := r.Text
			label := fmt.Sprintf("%s %.2f", textLabel, r.Confidence)
			labelWidth := d.MeasureString(label).Ceil()
			labelHeight := 12
			padding := 4

			labelY := y - labelHeight - padding*2
			if labelY < 0 {
				labelY = y + h + padding
			}

			bgRect := image.Rect(x, labelY, x+labelWidth+padding*2, labelY+labelHeight+padding*2)
			if bgRect.Max.X > bounds.Dx() {
				bgRect.Max.X = bounds.Dx()
			}
			if bgRect.Max.Y > bounds.Dy() {
				bgRect.Max.Y = bounds.Dy()
			}
			if bgRect.Min.Y < 0 {
				bgRect.Min.Y = 0
			}
			draw.Draw(img, bgRect, image.NewUniform(bboxColor), image.Point{}, draw.Over)
			d.Dot = fixed.P(x+padding, labelY+labelHeight+padding)
			d.DrawString(label)
		}
		if len(r.Mask) > 0 {
			color := colors[r.ClassID%int32(len(colors))]
			color.A = 100
			drawMask(img, r.Mask, &color)
		}
	}
}

func drawMask(img *image.RGBA, mask []float32, maskColor *color.RGBA) {
	slog.Debug("Drawing mask on image", "mask_size", len(mask))

	bounds := img.Bounds()

	if len(mask) != bounds.Dx()*bounds.Dy() {
		slog.Error("Mask size does not match image size", "mask_size", len(mask), "image_size", bounds.Dx()*bounds.Dy())
		return
	}

	var wg sync.WaitGroup
	workerSem := make(chan struct{}, runtime.NumCPU())
	for y := 0; y < bounds.Dy(); y++ {
		wg.Add(1)
		workerSem <- struct{}{}
		go func(y int) {
			defer wg.Done()
			defer func() { <-workerSem }()
			for x := 0; x < bounds.Dx(); x++ {
				idx := y*bounds.Dx() + x
				pixel := color.RGBAModel.Convert(img.At(x, y)).(color.RGBA)
				if maskColor != nil {
					alpha := mask[idx] * (float32(maskColor.A) / 255.0)
					invAlpha := 1.0 - alpha
					pixel.R = uint8(float32(maskColor.R)*alpha + float32(pixel.R)*invAlpha)
					pixel.G = uint8(float32(maskColor.G)*alpha + float32(pixel.G)*invAlpha)
					pixel.B = uint8(float32(maskColor.B)*alpha + float32(pixel.B)*invAlpha)
					pixel.A = 255
				} else {
					pixel.A = uint8(mask[idx] * 255)
				}
				img.Set(x, y, pixel)
			}
		}(y)
	}
	wg.Wait()
}

func CVPostProcess(input string, results []geniex_sdk.CVResult) (string, error) {
	file, err := os.Open(input)
	if err != nil {
		slog.Error("Failed to open image", "error", err)
		return "", err
	}
	defer file.Close()

	img, _, err := image.Decode(file)
	if err != nil {
		slog.Error("Failed to decode image", "error", err)
		return "", err
	}

	bounds := img.Bounds()
	rgba := image.NewRGBA(bounds)
	draw.Draw(rgba, bounds, img, bounds.Min, draw.Src)

	if len(results) == 1 && reflect.ValueOf(results[0].BBox).IsZero() {
		drawMask(rgba, results[0].Mask, nil)
	} else {
		drawBBoxes(rgba, results)
	}

	// save output image
	baseName := strings.TrimSuffix(filepath.Base(input), filepath.Ext(input))
	outputPath := filepath.Join(".", baseName+"_output.png")
	outFile, err := os.Create(outputPath)
	if err != nil {
		// Check if it's a permission error and provide a helpful message
		if os.IsPermission(err) {
			cwd, _ := os.Getwd()
			slog.Error("Permission denied writing output image", "output_path", outputPath, "cwd", cwd, "error", err)
			return "", fmt.Errorf("failed to write %s to the current directory. Permission denied", filepath.Base(outputPath))
		}
		slog.Error("Failed to create output image file", "error", err)
		return "", err
	}
	defer outFile.Close()

	err = png.Encode(outFile, rgba)
	if err != nil {
		slog.Error("Failed to encode output image", "error", err)
		return "", err
	}

	return outputPath, nil
}
