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

package model_hub

import (
	"context"
	"fmt"
	"io"
	"log/slog"
	"strings"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/service/s3"
	"github.com/aws/smithy-go/logging"
)

type Volces struct {
	s3Client *s3.Client
}

func NewVolces() *Volces {
	v := &Volces{}
	v.initS3Client()
	return v
}

var (
	errNotSupported = fmt.Errorf("not supported")
)

func (v *Volces) initS3Client() {
	cfg, err := config.LoadDefaultConfig(
		context.TODO(),
		config.WithLogger(logging.Nop{}),
		config.WithCredentialsProvider(aws.AnonymousCredentials{}),
		config.WithBaseEndpoint("https://tos-s3-cn-beijing.volces.com"),
		config.WithRegion("cn-beijing"),
	)
	if err != nil {
		panic("unable to load SDK config, " + err.Error())
	}
	v.s3Client = s3.NewFromConfig(cfg)
}

func (v *Volces) ChinaMainlandOnly() bool {
	return true
}

func (v *Volces) MaxConcurrency() int {
	return 4
}

func (v *Volces) CheckAvailable(ctx context.Context, modelName string) error {
	if !strings.HasPrefix(modelName, "NexaAI/") {
		return errNotSupported
	}

	modelName = strings.ReplaceAll(modelName, "NexaAI/", "model/") + "/"
	res, err := v.s3Client.ListObjectsV2(ctx, &s3.ListObjectsV2Input{
		Bucket:  aws.String("nexa-model-hub-bucket"),
		Prefix:  aws.String(modelName),
		MaxKeys: aws.Int32(1),
	})
	if err != nil {
		return err
	}

	if aws.ToInt32(res.KeyCount) == 0 {
		return fmt.Errorf("model %s not found", modelName)
	}

	return nil
}

func (v *Volces) ModelInfo(ctx context.Context, modelName string) ([]ModelFileInfo, error) {
	modelName = strings.ReplaceAll(modelName, "NexaAI/", "model/") + "/"

	data, err := v.s3Client.ListObjectsV2(ctx, &s3.ListObjectsV2Input{
		Bucket: aws.String("nexa-model-hub-bucket"),
		Prefix: aws.String(modelName),
	})
	if err != nil {
		return nil, err
	}

	res := make([]ModelFileInfo, len(data.Contents))
	for i, item := range data.Contents {
		res[i] = ModelFileInfo{
			Name: strings.TrimPrefix(*item.Key, modelName),
			Size: *item.Size,
		}
	}
	return res, nil
}

func (v *Volces) GetFileContent(ctx context.Context, modelName, fileName string, offset, limit int64, writer io.Writer) error {
	name := strings.ReplaceAll(modelName, "NexaAI/", "model/") + "/" + fileName

	slog.Debug("Volces GetFileContent", "modelName", modelName, "fileName", fileName, "name", name, "offset", offset, "limit", limit)

	input := &s3.GetObjectInput{
		Bucket: aws.String("nexa-model-hub-bucket"),
		Key:    aws.String(name),
	}

	if limit > 0 {
		input.Range = aws.String(fmt.Sprintf("bytes=%d-%d", offset, offset+limit-1))
	} else if offset > 0 {
		input.Range = aws.String(fmt.Sprintf("bytes=%d-", offset))
	}

	data, err := v.s3Client.GetObject(ctx, input)
	if err != nil {
		return err
	}
	defer data.Body.Close()

	_, err = io.Copy(writer, data.Body)
	return err
}
