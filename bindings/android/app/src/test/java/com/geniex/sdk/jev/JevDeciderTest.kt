// Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

package com.geniex.sdk.jev

import com.geniex.sdk.bean.DecideInput
import com.geniex.sdk.bean.DecideOutput
import kotlin.math.abs
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class JevDeciderTest {
    @Test
    fun classificationUsesNativeProbabilitiesAndPreservesMedia() = runBlocking {
        val requests = mutableListOf<DecideInput>()
        val decider = JevDecider(
            decide = { request ->
                requests += request
                Result.success(
                    DecideOutput(
                        probabilities = floatArrayOf(0.25f, 0.75f),
                        inputTokens = 12,
                        temperature = 0.5f,
                        decisionType = "lev"
                    )
                )
            },
            supportsMedia = true
        )

        val result = decider.classify(
            JevDecider.ClassificationSpec(
                labels = listOf("cat", "dog"),
                instructions = "Classify the animal.",
                context = "A friendly pet.",
                imagePaths = listOf("pet.png"),
                requestConfidence = true
            )
        ).getOrThrow()

        assertEquals("dog", result.label)
        assertEquals("sdk_calibrated", result.confidence?.provenance)
        assertFloatEquals(0.75f, result.confidence!!.value)
        assertEquals("A friendly pet.", requests.single().state)
        assertEquals(listOf("pet.png"), requests.single().imagePaths.toList())
        assertEquals(listOf("cat", "dog"), requests.single().questions.single().options.map { it.key })
    }

    @Test
    fun classificationSoftmaxesNativeLogits() = runBlocking {
        val decider = JevDecider(
            decide = {
                Result.success(
                    DecideOutput(
                        logits = floatArrayOf(0f, 2f),
                        temperature = 2f,
                        decisionType = "openjev"
                    )
                )
            },
            supportsMedia = false
        )

        val result = decider.classify(
            JevDecider.ClassificationSpec(
                labels = listOf("a", "b"),
                instructions = "Choose.",
                requestConfidence = true
            )
        ).getOrThrow()

        assertEquals("b", result.label)
        assertEquals("logits_softmax", result.confidence?.provenance)
        assertFloatEquals(0.7310586f, result.confidence!!.value)
    }

    @Test
    fun multipleChoiceReturnsFixedTopKAndRejectsRange() = runBlocking {
        val decider = JevDecider(
            decide = {
                Result.success(
                    DecideOutput(
                        logits = floatArrayOf(1f, 4f, 3f),
                        temperature = 1f,
                        decisionType = "generic"
                    )
                )
            },
            supportsMedia = false
        )
        val options = listOf(
            JevDecider.Option("a", "Alpha"),
            JevDecider.Option("b", "Beta"),
            JevDecider.Option("c", "Gamma")
        )

        val result = decider.choose(
            JevDecider.MultipleChoiceSpec(
                options = options,
                instructions = "Choose two.",
                minSelections = 2,
                maxSelections = 2,
                requestConfidence = true
            )
        ).getOrThrow()

        assertEquals(listOf("b", "c"), result.selectedIds)
        assertFloatEquals(0.25949646f, result.confidence!!.value)

        val failure = decider.choose(
            JevDecider.MultipleChoiceSpec(
                options = options,
                instructions = "Choose.",
                minSelections = 1,
                maxSelections = 2
            )
        ).exceptionOrNull()
        assertTrue(failure is IllegalArgumentException)
        assertTrue(failure!!.message!!.contains("fixed selection count"))
    }

    @Test
    fun triageRunsAllFiveNativeQuestions() = runBlocking {
        val outputs = ArrayDeque(
            listOf(
                DecideOutput(probabilities = floatArrayOf(0.8f, 0.05f, 0.04f, 0.03f, 0.04f, 0.04f)),
                DecideOutput(probabilities = floatArrayOf(0.9f, 0.1f)),
                DecideOutput(probabilities = floatArrayOf(0.05f, 0.15f, 0.7f, 0.1f)),
                DecideOutput(probabilities = floatArrayOf(0.85f, 0.15f)),
                DecideOutput(probabilities = floatArrayOf(0.2f, 0.8f))
            )
        )
        val indices = mutableListOf<Int>()
        val decider = JevDecider(
            decide = { request ->
                indices += request.questionIndex
                Result.success(outputs.removeFirst())
            },
            supportsMedia = true
        )

        val result = decider.triage(
            JevDecider.TriageSpec("I was charged twice and will leave.", listOf("bill.png"))
        ).getOrThrow()

        assertEquals(JevDecider.TriageResult("refund", "yes", "2", "yes", "no"), result)
        assertEquals(listOf(0, 1, 2, 3, 4), indices)
    }

    @Test
    fun textModelRejectsImagesBeforeNativeCall() = runBlocking {
        var calls = 0
        val decider = JevDecider(
            decide = {
                calls++
                Result.success(DecideOutput(probabilities = floatArrayOf(1f)))
            },
            supportsMedia = false
        )

        val failure = decider.classify(
            JevDecider.ClassificationSpec(
                labels = listOf("yes"),
                instructions = "Answer.",
                imagePaths = listOf("image.png")
            )
        ).exceptionOrNull()

        assertTrue(failure is IllegalArgumentException)
        assertEquals(0, calls)
    }

    private fun assertFloatEquals(expected: Float, actual: Float) {
        assertTrue("expected $expected, actual $actual", abs(expected - actual) < 0.00001f)
    }
}
