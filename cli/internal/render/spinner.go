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
	"os"
	"sync"

	"github.com/charmbracelet/huh/spinner"
)

type Spinner struct {
	*spinner.Spinner
	start sync.WaitGroup
	stop  sync.WaitGroup
}

func NewSpinner(desc string) *Spinner {
	s := Spinner{
		Spinner: spinner.New().Title(desc).Type(spinner.Globe),
	}

	s.Action(func() { s.start.Wait() })

	return &s
}

func (s *Spinner) Start() {
	s.start.Add(1)
	s.stop.Add(1)

	go func() {
		// if NO_COLOR is set, do not show spinner
		if os.Getenv("NO_COLOR") == "1" {
			fmt.Println(s.View())
			s.start.Wait()
		} else {
			s.Spinner.Run()
		}
		s.stop.Done()
	}()
}

func (s *Spinner) Stop() {
	s.start.Done()
	s.stop.Wait()
}
