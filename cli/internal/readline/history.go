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

package readline

// TODO: implement history persistence
type History struct {
	// config
	file string

	// state
	entries [][]rune
	index   int
}

func NewHistory(file string) *History {
	return &History{
		file:    file,
		entries: [][]rune{},
		index:   0,
	}
}

func (h *History) Add(entry []rune) {
	h.entries = append(h.entries, entry)
	h.index = len(h.entries)
}

func (h *History) Prev() []rune {
	if h.index > 0 {
		h.index--
		return h.entries[h.index]
	}
	return nil
}

func (h *History) Next() []rune {
	if h.index < len(h.entries)-1 {
		h.index++
		return h.entries[h.index]
	} else {
		h.index = len(h.entries)
		return []rune{}
	}
}

func (h *History) Save() error {
	return nil
}
