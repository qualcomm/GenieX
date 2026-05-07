# Android JNI Bridge Architecture

## Overview

Android bindings provide a Kotlin API for GenieX-Bridge using a JNI bridge pattern with 4 layers:

```
┌─────────────────────────────────────────────────────────┐
│ Layer 4: Public API (Kotlin)                          │
│ ModelWrapper.kt - Coroutine-based high-level API      │
│ (LlmWrapper, VlmWrapper, EmbedderWrapper, etc.)       │
└────────────────┬────────────────────────────────────────┘
                 │ calls
┌────────────────▼────────────────────────────────────────┐
│ Layer 3: JNI Interface (Kotlin)                        │
│ Model.kt - Declares external native methods           │
│ (Llm, Vlm, Embedder, Asr, Cv, Reranker)              │
└────────────────┬────────────────────────────────────────┘
                 │ JNI boundary
┌────────────────▼────────────────────────────────────────┐
│ Layer 2: JNI Bridge (C++)                             │
│ model_bridge_jni.cpp - Implements JNI methods         │
│ Java_com_geniex_sdk_jni_Model_* functions               │
└────────────────┬────────────────────────────────────────┘
                 │ calls C API
┌────────────────▼────────────────────────────────────────┐
│ Layer 1: Core Library (C)                             │
│ libgeniex_bridge.so - Provides ml_* API from ml.h       │
│ (ml_llm_create, ml_llm_generate, etc.)               │
└─────────────────────────────────────────────────────────┘
```

## Component Structure

Each AI capability (LLM, VLM, ASR, CV, Embedder, Reranker) follows this pattern:

### **LLM (Large Language Model) Example:**

#### Layer 4: **LlmWrapper.kt** (Public API)

- **Path:** `app/src/main/java/com/geniex/sdk/LlmWrapper.kt`
- **Purpose:** High-level Kotlin API with coroutine support
- **Features:** Builder pattern, Flow-based streaming, lifecycle management
- **Usage:** `LlmWrapper.builder().llmCreateInput(...).build()`

#### Layer 3: **Llm.kt** (JNI Interface)

- **Path:** `app/src/main/java/com/geniex/sdk/jni/Llm.kt`
- **Purpose:** Internal class declaring native JNI methods
- **Methods:**
  ```kotlin
  external fun create(llmCreateInputObj: LlmCreateInput): Long
  external fun destroy(handle: Long)
  external fun generate(handle: Long, prompt: String, config: GenerationConfig, cb: LLMTokenCallback): LlmGenerateResult
  external fun stopStream(handle: Long)
  external fun applyChatTemplate(handle: Long, messages: Array<ChatMessage>, tools: String?, enableThinking: Boolean): LlmApplyChatTemplateOutput
  ```

#### Layer 2: **llm_bridge_jni.cpp** (JNI Bridge)

- **Path:** `app/src/main/cpp/llm_bridge_jni.cpp`
- **Purpose:** Implements JNI methods, converts Java/Kotlin types to C types
- **JNI Functions:**
  - `Java_com_geniex_sdk_jni_Llm_create` → calls `ml_llm_create()`
  - `Java_com_geniex_sdk_jni_Llm_destroy` → calls `ml_llm_destroy()`
  - `Java_com_geniex_sdk_jni_Llm_generate` → calls `ml_llm_generate()`
  - `Java_com_geniex_sdk_jni_Llm_stopStream` → manages stop flags
  - `Java_com_geniex_sdk_jni_Llm_applyChatTemplate` → calls `ml_llm_apply_chat_template()`

#### Layer 1: **libgeniex_bridge.so** (Core Library)

- **Path:** `../../../build-android/out/libgeniex_bridge.so`
- **API Header:** `../../../include/ml.h`
- **Purpose:** C API providing ml*llm*\* functions
- **Built from:** `../../../src/*.cpp` (asr.cpp, llm.cpp, vlm.cpp, etc.)

### **Other Components** (Same 4-layer pattern):

| Component              | Wrapper            | JNI Interface | JNI Bridge              | Core API       |
| ---------------------- | ------------------ | ------------- | ----------------------- | -------------- |
| **Vision-Language**    | VlmWrapper.kt      | Vlm.kt        | vlm_bridge_jni.cpp      | ml*vlm*\*      |
| **Embeddings**         | EmbedderWrapper.kt | Embedder.kt   | embedder_bridge_jni.cpp | ml*embedder*\* |
| **Reranking**          | RerankerWrapper.kt | Reranker.kt   | reranker_bridge_jni.cpp | ml*reranker*\* |
| **Speech Recognition** | AsrWrapper.kt      | Asr.kt        | asr_bridge_jni.cpp      | ml*asr*\*      |
| **Computer Vision**    | CvWrapper.kt       | Cv.kt         | cv_bridge_jni.cpp       | ml*cv*\*       |

## Build System

### Native Library Compilation

- **Build Script:** `../../../build_android.sh`
- **CMake Entry:** `app/src/main/cpp/CMakeLists.txt`
- **Output:**
  - `libnpu_jni.so` (JNI bridge wrapper loaded by GeniexSdk.kt)
  - `libgeniex_bridge.so` (core ML library linked by JNI wrapper)
  - Plugin libraries: `libgeniex_plugin.so` (NPU backend for NPU acceleration)

### Library Loading

- **GeniexSdk.kt:** Initializes SDK and loads native libraries
  ```kotlin
  init {
      System.loadLibrary("npu_jni")  // Loads Layer 2 JNI bridge
  }
  ```

### Plugin Registration

- **NPU Backend:** Provides NPU acceleration for Qualcomm Snapdragon (using QNN library internally)
- **Path:** `../../../build-android/out/npu/`
- **Registration:** `GeniexSdk.init()` calls `registerPlugin()` to dynamically load plugin libraries

## Directory Structure

```
bindings/android/
├── app/
│   ├── src/main/
│   │   ├── cpp/                      # Layer 2: JNI Bridge (C++)
│   │   │   ├── llm_bridge_jni.cpp
│   │   │   ├── vlm_bridge_jni.cpp
│   │   │   ├── embedder_bridge_jni.cpp
│   │   │   ├── reranker_bridge_jni.cpp
│   │   │   ├── asr_bridge_jni.cpp
│   │   │   ├── cv_bridge_jni.cpp
│   │   │   ├── geniex_sdk.cpp          # SDK initialization & plugin registration
│   │   │   ├── jniutils.cpp/.h       # JNI type conversion utilities
│   │   │   ├── jni_cb.cpp/.h         # Callback handling for streaming
│   │   │   ├── qnn/CMakeLists.txt    # NPU backend build config (QNN library)
│   │   │   └── CMakeLists.txt
│   │   └── java/com/geniex/sdk/
│   │       ├── jni/                  # Layer 3: JNI Interface (Kotlin)
│   │       │   ├── Llm.kt
│   │       │   ├── Vlm.kt
│   │       │   ├── Embedder.kt
│   │       │   ├── Reranker.kt
│   │       │   ├── Asr.kt
│   │       │   └── Cv.kt
│   │       ├── LlmWrapper.kt         # Layer 4: Public API
│   │       ├── VlmWrapper.kt
│   │       ├── EmbedderWrapper.kt
│   │       ├── RerankerWrapper.kt
│   │       ├── GeniexSdk.kt            # SDK entry point
│   │       └── bean/                 # Data classes
│   └── build.gradle.kts              # Build configuration
└── README.md                         # This file

../../include/ml.h                    # Layer 1: Core C API header
../../src/*.cpp                       # Layer 1: Core implementation
../../build-android/out/
├── libgeniex_bridge.so                 # Layer 1: Core library
└── npu/                              # NPU plugin libraries (QNN-based)
    └── libgeniex_plugin.so
```

## Publish to Maven

### Preparations for Releasing Android SDK aar

Generating the repo folder in maven format.

Under `GenieX-bridge/bindings/android`:

1. Modify `tmpVersion` in `app/update.gradle` to the new version.
2. Sync project with Gradle files.
3. Execute `./gradlew assembleRelease`.
4. Execute `./gradlew publish` to obtain the `repo` folder.

### Publishing to Maven Central

TODO: update the Maven Central publishing procedure for GenieX-Bridge.

### Publishing to GitHub

TODO: update the GitHub-based Android artifact publishing instructions for GenieX-Bridge.

## Logging

SDK logs are routed to `logcat` under the tag `GeniexSdk` at the corresponding
Android priority (`VERBOSE`/`DEBUG`/`INFO`/`WARN`/`ERROR`). Set the runtime
level from Kotlin:

```kotlin
GeniexSdk.setLogLevel("debug")  // trace|debug|info|warn|error|none
```

`GeniexSdk.setLogLevel` takes precedence over the `GENIEX_LOG` environment
variable (which is rarely settable inside an Android process).

```bash
adb logcat -s GeniexSdk
```
