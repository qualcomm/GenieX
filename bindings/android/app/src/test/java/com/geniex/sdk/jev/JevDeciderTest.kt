// Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause

package com.geniex.sdk.jev

import com.geniex.sdk.bean.LlmStreamResult
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertSame
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlinx.coroutines.flow.flowOf
import kotlinx.coroutines.runBlocking

class JevDeciderTest {
    @Test
    fun acceptsOneFencedObjectWithExactFields() {
        val result = JevDecider.decodeExactObject(
            "```json\n{\"label\":\"yes\"}\n```",
            setOf("label")
        )

        assertEquals("yes", result.getString("label"))
    }

    @Test
    fun rejectsDuplicateNestedAndEscapedKeysOnlyAtObjectLevel() {
        assertFalse(JevDecider.noDuplicateFields("{\"label\":\"one\",\"label\":\"two\"}"))
        assertTrue(JevDecider.noDuplicateFields("{\"label\":\"quoted \\\"field\\\": value\"}"))
        assertTrue(JevDecider.noDuplicateFields("{\"label\":{\"label\":\"nested\"}}"))
    }

    @Test
    fun rejectsWrongObjectShapeAndTrailingText() {
        assertFailsWithIllegalArgument {
            JevDecider.decodeExactObject("{\"label\":\"yes\",\"extra\":true}", setOf("label"))
        }
        assertFailsWithIllegalArgument {
            JevDecider.decodeExactObject("{\"label\":\"yes\"} prose", setOf("label"))
        }
        assertFailsWithIllegalArgument {
            JevDecider.decodeExactObject("{\"label\":\"yes\"}{\"label\":\"no\"}", setOf("label"))
        }
    }

    @Test
    fun llamaGrammarUsesActualRuleSeparators() {
        val companion = JevDecider::class.java.getField("Companion").get(null)
        val grammarMethod = companion.javaClass.getDeclaredMethod(
            "classificationGrammar",
            List::class.java,
            Boolean::class.javaPrimitiveType
        )
        grammarMethod.isAccessible = true

        val grammar = grammarMethod.invoke(companion, listOf("yes"), false) as String

        assertTrue(grammar.contains("ws \"}\"\nlabel ::= "))
        assertFalse(grammar.contains("ws \"}\"\\nlabel ::= "))
    }

    @Test
    fun generatedTextJoinsStreamTokens() = runBlocking {
        val text = JevDecider.collectGeneratedText(
            flowOf(LlmStreamResult.Token("{\"label\":"), LlmStreamResult.Token("\"yes\"}"))
        )

        assertEquals("{\"label\":\"yes\"}", text)
    }

    @Test
    fun generatedTextPropagatesStreamErrors() = runBlocking {
        val failure = IllegalStateException("generation failed")

        try {
            JevDecider.collectGeneratedText(flowOf(LlmStreamResult.Error(failure)))
        } catch (actual: IllegalStateException) {
            assertSame(failure, actual)
            return@runBlocking
        }
        throw AssertionError("expected stream failure")
    }

    private fun assertFailsWithIllegalArgument(block: () -> Unit) {
        try {
            block()
        } catch (_: IllegalArgumentException) {
            return
        }
        throw AssertionError("expected IllegalArgumentException")
    }
}
