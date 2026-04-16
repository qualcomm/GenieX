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

package render

import (
	"fmt"
	"testing"
)

func TestThemeColor(t *testing.T) {
	theme := GetTheme()
	fmt.Println(theme.Normal.Sprint("This is a normal message."))
	fmt.Println(theme.Info.Sprint("This is an info message."))
	fmt.Println(theme.Success.Sprint("This is a success message."))
	fmt.Println(theme.Warning.Sprint("This is a warning message."))
	fmt.Println(theme.Error.Sprint("This is an error message."))

	fmt.Println(theme.Quant.Sprint("🔹 Quant=Q4_K_M"))
	fmt.Println(theme.Prompt.Sprint("> ") + "/path/to/image how are you")
	fmt.Println(theme.AddFiles.Sprint("add image: /path/to/image"))
	fmt.Println(theme.ThinkOutput.Sprint("<think>this is think message</think>"))
	fmt.Println(theme.ModelOutput.Sprint("hi how are you today"))
	fmt.Println(theme.Profile.Sprint("— 64.8 tok/s • 61 tok • 18.4 s first token —"))
}
