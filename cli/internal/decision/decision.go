// Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

// Package decision contains the model-independent request and scoring logic
// shared by the local CLI and HTTP handler.
package decision

import (
	"errors"
	"fmt"
	"math"
	"slices"
	"strconv"

	geniex_sdk "github.com/qualcomm/GenieX/bindings/go"
)

type Choice struct {
	Value       any
	Description string
}

type Level struct {
	Label       string
	Description string
}

// Kind uses public Decisions names; the SDK maps predicate to systemone's noul.
type Kind string

const (
	KindPredicate Kind = "predicate"
	KindChoice    Kind = "choice"
	KindScore     Kind = "score"
)

type Question struct {
	ID           string
	Kind         Kind
	Instructions string
	Choices      []Choice
	Levels       []Level
}

func candidateCount(question Question) (int, error) {
	switch question.Kind {
	case KindPredicate:
		if len(question.Choices) != 2 {
			return 0, errors.New("predicate requires two candidates")
		}
		seen := map[bool]bool{}
		for _, choice := range question.Choices {
			value, ok := choice.Value.(bool)
			if !ok || seen[value] {
				return 0, errors.New("predicate candidates must be true and false")
			}
			seen[value] = true
		}
		return len(question.Choices), nil
	case KindChoice:
		if len(question.Choices) < 2 || len(question.Choices) > 255 {
			return 0, errors.New("choice requires 2–255 candidates")
		}
		return len(question.Choices), nil
	case KindScore:
		if len(question.Levels) < 2 || len(question.Levels) > 10 {
			return 0, errors.New("score requires 2–10 levels")
		}
		return len(question.Levels), nil
	default:
		return 0, fmt.Errorf("unsupported decision kind %q", question.Kind)
	}
}

// Questions validates and converts public questions to SDK options. Model-specific
// labels and ordering are added by the SDK from the GGUF decision metadata.
func Questions(questions []Question) ([]geniex_sdk.DecideQuestion, error) {
	if len(questions) == 0 {
		return nil, errors.New("questions must not be empty")
	}
	encoded := make([]geniex_sdk.DecideQuestion, len(questions))
	for i, question := range questions {
		if _, err := candidateCount(question); err != nil {
			return nil, fmt.Errorf("question %d: %w", i, err)
		}
		entry := geniex_sdk.DecideQuestion{
			ID: question.ID, Type: string(question.Kind), Instructions: question.Instructions,
		}
		switch question.Kind {
		case KindPredicate, KindChoice:
			for _, choice := range question.Choices {
				option := geniex_sdk.DecideOption{}
				switch value := choice.Value.(type) {
				case string:
					option.Key = value
				case bool:
					// Predicate keys stay strings; choice booleans retain their JSON type in Jinja.
					option.Key = strconv.FormatBool(value)
					option.KeyIsBool = question.Kind == KindChoice
				default:
					return nil, fmt.Errorf("question %d: choice value must be a string or boolean", i)
				}
				if choice.Description != "" {
					option.Description = &choice.Description
				}
				entry.Options = append(entry.Options, option)
			}
		case KindScore:
			for levelIndex, level := range question.Levels {
				entry.Options = append(entry.Options, geniex_sdk.DecideOption{
					Key: strconv.Itoa(levelIndex), Description: &level.Description,
				})
			}
		}
		encoded[i] = entry
	}
	return encoded, nil
}

type Probability struct {
	Value       any     `json:"value"`
	Label       *string `json:"label,omitempty"`
	Probability float64 `json:"probability"`
}

type Result struct {
	Probability   *float64      `json:"probability,omitempty"`
	Choice        any           `json:"choice,omitempty"`
	Probabilities []Probability `json:"probabilities,omitempty"`
	Confidence    *float64      `json:"confidence,omitempty"`
	Score         *float64      `json:"score,omitempty"`
}

// Measure how far the winning choice exceeds uniform guessing.
func choiceConfidence(probabilities []float64) float64 {
	if len(probabilities) < 2 {
		return 1
	}
	uniform := 1 / float64(len(probabilities))
	peak := slices.Max(probabilities)
	return max(0, (peak-uniform)/(1-uniform))
}

// Compare expected distance from the modal level against a uniform distribution.
func scoreConfidence(probabilities []float64) float64 {
	if len(probabilities) < 2 {
		return 1
	}
	mode := 0
	for i := 1; i < len(probabilities); i++ {
		if probabilities[i] > probabilities[mode] {
			mode = i
		}
	}
	distance, uniformDistance := 0.0, 0.0
	center := float64(len(probabilities)-1) / 2
	for i, probability := range probabilities {
		distance += probability * math.Abs(float64(i-mode))
		uniformDistance += math.Abs(float64(i)-center) / float64(len(probabilities))
	}
	if uniformDistance == 0 {
		return 1
	}
	return max(0, 1-distance/uniformDistance)
}

// Answer applies the temperature read from the model's GGUF metadata and
// GenieX's model-independent TypeSafe confidence formulas.
func Answer(question Question, logits []float32, temperature float64) (Result, error) {
	count, err := candidateCount(question)
	if err != nil {
		return Result{}, err
	}
	if len(logits) != count {
		return Result{}, errors.New("wrong number of candidate logits")
	}
	if math.IsNaN(temperature) || math.IsInf(temperature, 0) || temperature <= 0 {
		return Result{}, errors.New("decision temperature must be finite and positive")
	}

	// Subtract the maximum logit to avoid overflow before normalizing.
	peak := math.Inf(-1)
	for _, logit := range logits {
		if math.IsNaN(float64(logit)) || math.IsInf(float64(logit), 0) {
			return Result{}, errors.New("non-finite candidate logit")
		}
		peak = max(peak, float64(logit))
	}
	probabilities := make([]float64, len(logits))
	var sum float64
	for i, logit := range logits {
		probabilities[i] = math.Exp((float64(logit) - peak) / temperature)
		sum += probabilities[i]
	}
	for i := range probabilities {
		probabilities[i] /= sum
	}
	return resultFromProbabilities(question, probabilities)
}

// AnswerProbabilities formats calibrated SDK probabilities without applying temperature twice.
func AnswerProbabilities(question Question, values []float32) (Result, error) {
	count, err := candidateCount(question)
	if err != nil {
		return Result{}, err
	}
	if len(values) != count {
		return Result{}, errors.New("wrong number of candidate probabilities")
	}
	probabilities := make([]float64, count)
	var sum float64
	for i, value := range values {
		probability := float64(value)
		if math.IsNaN(probability) || math.IsInf(probability, 0) || probability < 0 {
			return Result{}, errors.New("invalid candidate probability")
		}
		probabilities[i] = probability
		sum += probability
	}
	if sum == 0 {
		return Result{}, errors.New("candidate probabilities must have positive sum")
	}
	for i := range probabilities {
		probabilities[i] /= sum
	}
	return resultFromProbabilities(question, probabilities)
}

// Map either logits-derived or SDK-calibrated probabilities to the public answer shape.
func resultFromProbabilities(question Question, probabilities []float64) (Result, error) {
	if question.Kind == KindPredicate {
		trueIndex := 0
		if !question.Choices[0].Value.(bool) {
			trueIndex = 1
		}
		probability := probabilities[trueIndex]
		return Result{Probability: &probability}, nil
	}

	confidence := choiceConfidence(probabilities)
	result := Result{
		Confidence:    &confidence,
		Probabilities: make([]Probability, len(probabilities)),
	}
	if question.Kind == KindChoice {
		winner := 0
		for i, choice := range question.Choices {
			if probabilities[i] > probabilities[winner] {
				winner = i
			}
			result.Probabilities[i] = Probability{Value: choice.Value, Probability: probabilities[i]}
		}
		result.Choice = question.Choices[winner].Value
		return result, nil
	}

	var score float64
	for i, level := range question.Levels {
		label := level.Label
		result.Probabilities[i] = Probability{Value: i, Label: &label, Probability: probabilities[i]}
		score += float64(i) * probabilities[i]
	}
	confidence = scoreConfidence(probabilities)
	result.Confidence = &confidence
	result.Score = &score
	return result, nil
}
