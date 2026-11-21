// Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

package com.geniex.sdk.jev

import com.geniex.sdk.LlmWrapper
import com.geniex.sdk.VlmWrapper
import com.geniex.sdk.bean.DecideInput
import com.geniex.sdk.bean.DecideOption
import com.geniex.sdk.bean.DecideOutput
import com.geniex.sdk.bean.DecideQuestion
import kotlin.math.exp

/** Typed, non-executing decisions backed by the SDK logits path. */
class JevDecider internal constructor(
    private val decide: suspend (DecideInput) -> Result<DecideOutput>,
    private val supportsMedia: Boolean,
) {
    data class Confidence(val value: Float, val provenance: String)

    data class ClassificationSpec(
        val labels: List<String>,
        val instructions: String,
        val context: String = "",
        val imagePaths: List<String> = emptyList(),
        val requestConfidence: Boolean = false,
    )

    data class ClassificationResult(val label: String, val confidence: Confidence? = null)
    data class Option(val id: String, val text: String)

    data class MultipleChoiceSpec(
        val options: List<Option>,
        val instructions: String,
        val context: String = "",
        val imagePaths: List<String> = emptyList(),
        val minSelections: Int = 0,
        val maxSelections: Int = 0,
        val requestConfidence: Boolean = false,
    )

    data class MultipleChoiceResult(val selectedIds: List<String>, val confidence: Confidence? = null)
    data class TriageSpec(val message: String, val imagePaths: List<String> = emptyList())
    data class TriageResult(
        val intent: String,
        val isUrgent: String,
        val frustration: String,
        val refundRequested: String,
        val churnRisk: String,
    )

    suspend fun classify(spec: ClassificationSpec): Result<ClassificationResult> = runCatching {
        val labels = validateIds(spec.labels, "classification labels")
        require(spec.instructions.isNotBlank()) { "classification instructions are required" }
        val question = DecideQuestion(
            id = "classification",
            questionType = "choice",
            instructions = spec.instructions,
            options = labels.map { DecideOption(it) }.toTypedArray(),
        )
        val output = runDecision(spec.context, arrayOf(question), 0, spec.imagePaths)
        val (probabilities, provenance) = probabilities(output, labels.size)
        val selected = probabilities.indices.maxBy { probabilities[it] }
        ClassificationResult(
            labels[selected],
            if (spec.requestConfidence) Confidence(probabilities[selected], provenance) else null,
        )
    }

    suspend fun choose(spec: MultipleChoiceSpec): Result<MultipleChoiceResult> = runCatching {
        val ids = validateIds(spec.options.map { it.id }, "multiple-choice option IDs")
        require(spec.instructions.isNotBlank()) { "multiple-choice instructions are required" }
        val count = selectionCount(spec.minSelections, spec.maxSelections, ids.size)
        val question = DecideQuestion(
            id = "selection",
            questionType = "choice",
            instructions = spec.instructions,
            options = spec.options.map { DecideOption(it.id, description = it.text) }.toTypedArray(),
        )
        val output = runDecision(spec.context, arrayOf(question), 0, spec.imagePaths)
        val (probabilities, provenance) = probabilities(output, ids.size)
        val selected = probabilities.indices.sortedByDescending { probabilities[it] }.take(count)
        MultipleChoiceResult(
            selected.map { ids[it] },
            if (spec.requestConfidence) Confidence(selected.minOf { probabilities[it] }, provenance) else null,
        )
    }

    suspend fun triage(spec: TriageSpec): Result<TriageResult> = runCatching {
        require(spec.message.isNotBlank()) { "triage message is required" }
        val questions = triageQuestions()
        val values = questions.indices.map { index ->
            val output = runDecision(spec.message, questions, index, spec.imagePaths)
            val (scores) = probabilities(output, questions[index].options.size)
            questions[index].options[scores.indices.maxBy { scores[it] }].key
        }
        TriageResult(values[0], yesNo(values[1]), values[2], yesNo(values[3]), yesNo(values[4]))
    }

    private suspend fun runDecision(
        state: String,
        questions: Array<DecideQuestion>,
        questionIndex: Int,
        images: List<String>,
    ): DecideOutput {
        require(supportsMedia || images.isEmpty()) { "text structured decisions cannot include images" }
        return decide(
            DecideInput(
                state = state,
                questions = questions,
                questionIndex = questionIndex,
                imagePaths = images.toTypedArray(),
            )
        ).getOrThrow()
    }

    companion object {
        private val TRIAGE_INTENTS = listOf(
            "refund",
            "technical_help",
            "billing_question",
            "information",
            "cancellation",
            "other",
        )

        @JvmStatic
        fun forLlm(wrapper: LlmWrapper): JevDecider = JevDecider(wrapper::decide, supportsMedia = false)

        @JvmStatic
        fun forVlm(wrapper: VlmWrapper): JevDecider = JevDecider(wrapper::decide, supportsMedia = true)

        private fun probabilities(output: DecideOutput, expected: Int): Pair<FloatArray, String> {
            output.probabilities?.let {
                require(it.size == expected) { "SDK decision returned an unexpected probability count" }
                return it.copyOf() to "sdk_calibrated"
            }
            val logits = output.logits
            require(logits != null && logits.size == expected) { "SDK decision returned an unexpected logit count" }
            require(output.temperature > 0f && output.temperature.isFinite()) {
                "SDK decision returned an invalid temperature"
            }
            val peak = logits.max()
            val weights = DoubleArray(logits.size) { exp((logits[it] - peak) / output.temperature.toDouble()) }
            val total = weights.sum()
            require(total > 0.0 && total.isFinite()) { "SDK decision returned invalid logits" }
            return FloatArray(weights.size) { (weights[it] / total).toFloat() } to "logits_softmax"
        }

        private fun validateIds(values: List<String>, label: String): List<String> {
            require(values.isNotEmpty()) { "$label are required" }
            require(values.all { it.isNotBlank() } && values.distinct().size == values.size) {
                "$label must be non-empty and unique"
            }
            return values
        }

        private fun selectionCount(minimum: Int, maximum: Int, optionCount: Int): Int {
            if (minimum == 0 && maximum == 0) return 1
            require(minimum == maximum) { "native decisions require a fixed selection count" }
            require(minimum in 1..optionCount) { "invalid selection count" }
            return minimum
        }

        private fun yesNo(value: String): String = if (value == "true") "yes" else "no"

        private fun triageQuestions(): Array<DecideQuestion> {
            val predicates = arrayOf(DecideOption("true"), DecideOption("false"))
            return arrayOf(
                DecideQuestion(
                    "intent",
                    "choice",
                    "Classify the customer message intent.",
                    TRIAGE_INTENTS.map { DecideOption(it) }.toTypedArray(),
                ),
                DecideQuestion(
                    "is_urgent",
                    "predicate",
                    "Does the message communicate time pressure or a deadline?",
                    predicates.copyOf(),
                ),
                DecideQuestion(
                    "frustration",
                    "score",
                    "Score frustration from calm and neutral to very angry or strong language.",
                    arrayOf(
                        DecideOption("0", description = "calm and neutral"),
                        DecideOption("1", description = "concerned but civil"),
                        DecideOption("2", description = "clearly annoyed"),
                        DecideOption("3", description = "very angry or strong language"),
                    ),
                ),
                DecideQuestion(
                    "refund_requested",
                    "predicate",
                    "Does the customer ask for money back?",
                    predicates.copyOf(),
                ),
                DecideQuestion(
                    "churn_risk",
                    "predicate",
                    "Does the message suggest leaving, cancelling, or using a competitor?",
                    predicates.copyOf(),
                ),
            )
        }
    }
}
