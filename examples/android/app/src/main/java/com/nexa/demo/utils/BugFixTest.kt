// Copyright 2024-2026 Nexa AI, Inc.
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

//package com.nexa.demo.utils
//
//import com.nexa.sdk.LlmWrapper
//import com.nexa.sdk.bean.ChatMessage
//import com.nexa.sdk.bean.LlmApplyChatTemplateOutput
//import com.nexa.sdk.jni.LLmExt
//import kotlinx.coroutines.Dispatchers
//import kotlinx.coroutines.withContext
//import kotlin.reflect.full.memberProperties
//import kotlin.reflect.jvm.isAccessible
//
//class BugFixTest {
//
//    companion object {
//        suspend fun applyChatTemplate(
//            lLmExt: LLmExt,
//            llmWrapper: LlmWrapper,
//            messages: Array<ChatMessage>,
//            tools: String?,
//            enableThinking: Boolean
//        ): Result<LlmApplyChatTemplateOutput> =
//            withContext(Dispatchers.IO) {
//                var handle = 0L
//                llmWrapper::class.memberProperties.forEach {
//                    if (it.name == "handle") {
//                        it.isAccessible = true
//                        handle = it.call(llmWrapper) as Long
//                        return@forEach
//                    }
//                }
//                if (handle == 0L) {
//                    return@withContext Result.failure(IllegalStateException("LLM not initialized"))
//                }
//                runCatching {
//                    lLmExt.applyChatTemplateExt(handle, messages, tools, enableThinking)
//                }
//            }
//    }
//}