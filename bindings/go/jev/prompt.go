// Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

package jev

import (
	"encoding/json"
	"fmt"
)

const systemPrompt = `You are GenieX JEV, a careful local visual browser agent.
Complete the user's task one browser action at a time. You receive a current screenshot and an indexed list of visible controls. Select controls only by their listed numeric index. Never invent selectors, JavaScript, browser commands, or an element not in the list.
Return exactly one JSON object and no markdown or explanation. The object has an "action" field. Supported actions are navigate, click, type, scroll, wait, read, go_back, go_forward, and finish. Include only fields required by the selected action. Finish only when the task is complete and include final_answer.`

// SystemPrompt is the fixed instruction applied to every decision.
func SystemPrompt() string { return systemPrompt }

// Grammar is intentionally narrow enough for llama.cpp constrained decoding.
// Host validation remains mandatory because grammar cannot prove target freshness.
const Grammar = `root ::= object
object ::= "{" ws "\"action\"" ws ":" ws action-fields ws "}"
action-fields ::= navigate | click | type | scroll | wait | read | simple | finish
navigate ::= "\"navigate\"" ws "," ws "\"url\"" ws ":" ws string
click ::= "\"click\"" ws "," ws "\"index\"" ws ":" ws integer
type ::= "\"type\"" ws "," ws "\"index\"" ws ":" ws integer ws "," ws "\"text\"" ws ":" ws string (ws "," ws "\"submit\"" ws ":" ws boolean)?
scroll ::= "\"scroll\"" ws "," ws "\"direction\"" ws ":" ws ("\"up\"" | "\"down\"") ws "," ws "\"amount\"" ws ":" ws integer
wait ::= "\"wait\"" ws "," ws "\"milliseconds\"" ws ":" ws integer
read ::= "\"read\"" ws "," ws "\"index\"" ws ":" ws integer
simple ::= "\"go_back\"" | "\"go_forward\""
finish ::= "\"finish\"" ws "," ws "\"final_answer\"" ws ":" ws string
boolean ::= "true" | "false"
integer ::= [0-9]+
string ::= "\"" chars "\""
chars ::= ([^"\\] | "\\" (["\\/bfnrt] | "u" [0-9a-fA-F]{4}))*
ws ::= [ \t\n\r]*`

// RoundPrompt renders the compact trusted text supplied with the screenshot.
func RoundPrompt(task string, observation Observation, ledger []string, correction string) (string, error) {
	elements, err := json.Marshal(observation.Elements)
	if err != nil {
		return "", fmt.Errorf("marshal observation elements: %w", err)
	}
	prompt := fmt.Sprintf("Task: %s\nCurrent URL: %s\nPage title: %s\nVisible elements: %s", task, observation.URL, observation.Title, elements)
	if len(ledger) > 0 {
		prompt += "\nCompleted steps:"
		for _, step := range ledger {
			prompt += "\n- " + step
		}
	}
	if correction != "" {
		prompt += "\nPrevious response was rejected: " + correction + ". Return a valid action for this same observation."
	}
	return prompt, nil
}
