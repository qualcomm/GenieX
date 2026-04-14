<div align="center" style="text-decoration: none;">
  <img width="100%" src="assets/banner1.png" alt="Nexa AI Banner">
  <p>
    <a href="https://www.producthunt.com/products/nexasdk-for-mobile?embed=true&utm_source=badge-top-post-badge&utm_medium=badge&utm_campaign=badge-nexasdk-for-mobile" target="_blank" rel="noopener noreferrer">
        <img alt="NexaSDK for Mobile - #1 Product of the Day" width="180" height="39" src="https://api.producthunt.com/widgets/embed-image/v1/top-post-badge.svg?post_id=1049998&theme=dark&period=daily&t=1765991451976">
    </a>
    <a href="https://trendshift.io/repositories/12239" target="_blank" rel="noopener noreferrer">
        <img alt="NexaAI/nexa-sdk - #1 Repository of the Day" height="39" src="https://trendshift.io/api/badge/repositories/12239">
    </a>
  </p>
  <p>
    <a href="https://docs.nexa.ai">
        <img src="https://img.shields.io/badge/docs-website-brightgreen?logo=readthedocs" alt="Documentation">
    </a>
  </p>
</div>

# NexaSDK

**NexaSDK lets you build the smartest and fastest on-device AI with minimum energy.** It is a highly performant local inference framework that runs the latest multimodal AI models locally on NPU, GPU, and CPU - across Android, Windows, and Linux devices with a few lines of code.

NexaSDK supported latest models **weeks or months before anyone else** — Qwen3-VL, DeepSeek-OCR, Gemma3n (Vision), and more.

> ⭐ **Star this repo** to keep up with exciting updates and new releases about latest on-device AI capabilities.

## 🏆 Recognized Milestones

- **Qualcomm** featured us **3 times** in official blogs.
  - [Innovating Multimodal AI on Qualcomm Hexagon NPU](https://www.qualcomm.com/developer/blog/2025/09/omnineural-4b-nexaml-qualcomm-hexagon-npu).
  - [First-ever Day-0 model support on Qualcomm Hexagon NPU for compute and mobile platforms, Auto and IoT](https://www.qualcomm.com/developer/blog/2025/10/granite-4-0-to-the-edge-on-device-ai-for-real-world-performance).
  - [A simple way to bring on-device AI to smartphones with Snapdragon](https://www.qualcomm.com/developer/blog/2025/11/nexa-ai-for-android-simple-way-to-bring-on-device-ai-to-smartphones-with-snapdragon)

## 🚀 Quick Start

| Platform        | Links                                                                                     |
| --------------- | ----------------------------------------------------------------------------------------- |
| 🖥️ CLI          | [Quick Start](#-cli) ｜ [Docs](https://docs.nexa.ai/en/nexa-sdk-go/NexaCLI)               |
| 🐍 Python       | [Quick Start](#-python-sdk) ｜ [Docs](https://docs.nexa.ai/en/nexa-sdk-python/overview)   |
| 🤖 Android      | [Quick Start](#-android-sdk) ｜ [Docs](https://docs.nexa.ai/en/nexa-sdk-android/overview) |
| 🐳 Linux Docker | [Quick Start](#-linux-docker) ｜ [Docs](https://docs.nexa.ai/en/nexa-sdk-docker/overview) |

---

### 🖥️ CLI

**Download:**

| Windows                                                                                                  | Linux                                                                                        |
| -------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| [arm64 (Qualcomm NPU)](https://public-storage.nexa4ai.com/nexa_sdk/downloads/nexa-cli_windows_arm64.exe) | [arm64](https://github.com/NexaAI/nexa-sdk/releases/latest/download/nexa-cli_linux_arm64.sh) |
| [x64](https://public-storage.nexa4ai.com/nexa_sdk/downloads/nexa-cli_windows_x86_64.exe) | [x64](https://github.com/NexaAI/nexa-sdk/releases/latest/download/nexa-cli_linux_x86_64.sh)  |

**Run your first model:**

```bash
# Chat with Qwen3
nexa infer ggml-org/Qwen3-1.7B-GGUF

# Multimodal: drag images into the CLI
nexa infer NexaAI/Qwen3-VL-4B-Instruct-GGUF

# NPU (Windows arm64 with Snapdragon X Elite)
nexa infer NexaAI/OmniNeural-4B
```

**NPU Access Token (required for NPU models):**

> **Note:** Our previous token validation service has been deprecated. For any NPU usage, simply set the access token below — no additional registration or validation is needed.

For Windows:
```shell
$env:NEXA_TOKEN="key/eyJhY2NvdW50Ijp7ImlkIjoiNDI1Y2JiNWQtNjk1NC00NDYxLWJiOWMtYzhlZjBiY2JlYzA2In0sInByb2R1Y3QiOnsiaWQiOiJkYjI4ZTNmYy1mMjU4LTQ4ZTctYmNkYi0wZmE4YjRkYTJhNWYifSwicG9saWN5Ijp7ImlkIjoiMmYyOWQyMjctNDVkZS00MzQ3LTg0YTItMjUwNTYwMmEzYzMyIiwiZHVyYXRpb24iOjMxMTA0MDAwMH0sInVzZXIiOnsiaWQiOiI3MGE2YzA4NS1jYjc3LTQ3YmEtOWUxNC1lNjFjYTA2ZThmZjUiLCJlbWFpbCI6ImFsYW40QG5leGE0YWkuY29tIn0sImxpY2Vuc2UiOnsiaWQiOiI4OTlhZGQ2NS1lOTI2LTQ2M2ItODllNi0xMjc0NzM3ZjA1MzYiLCJjcmVhdGVkIjoiMjAyNS0wOS0wNlQwMDo1MzozNi4yMDNaIiwiZXhwaXJ5IjoiMjAzNS0xMi0zMVQyMzo1OTo1OS4wMDBaIn19.BXoUHIEzFMuuZbBT7RvsKO9nTi5950C6kHO64blF7XBnfKvZ6ClA8a55tmszI1ZWdngzpNFTzMM5PV5euuzMCA=="
```

For Linux / Android adb shell:
```shell
export NEXA_TOKEN="key/eyJhY2NvdW50Ijp7ImlkIjoiNDI1Y2JiNWQtNjk1NC00NDYxLWJiOWMtYzhlZjBiY2JlYzA2In0sInByb2R1Y3QiOnsiaWQiOiJkYjI4ZTNmYy1mMjU4LTQ4ZTctYmNkYi0wZmE4YjRkYTJhNWYifSwicG9saWN5Ijp7ImlkIjoiMmYyOWQyMjctNDVkZS00MzQ3LTg0YTItMjUwNTYwMmEzYzMyIiwiZHVyYXRpb24iOjMxMTA0MDAwMH0sInVzZXIiOnsiaWQiOiI3MGE2YzA4NS1jYjc3LTQ3YmEtOWUxNC1lNjFjYTA2ZThmZjUiLCJlbWFpbCI6ImFsYW40QG5leGE4YWkuY29tIn0sImxpY2Vuc2UiOnsiaWQiOiI4OTlhZGQ2NS1lOTI2LTQ2M2ItODllNi0xMjc0NzM3ZjA1MzYiLCJjcmVhdGVkIjoiMjAyNS0wOS0wNlQwMDo1MzozNi4yMDNaIiwiZXhwaXJ5IjoiMjAzNS0xMi0zMVQyMzo1OTo1OS4wMDBaIn19.BXoUHIEzFMuuZbBT7RvsKO9nTi5950C6kHO64blF7XBnfKvZ6ClA8a55tmszI1ZWdngzpNFTzMM5PV5euuzMCA=="
```

- **Models:** LLM, Multimodal, ASR, OCR, Rerank, Object Detection, Image Generation, Embedding
- **Formats:** GGUF, NEXA
- 📖 [CLI Reference Docs](https://docs.nexa.ai/en/nexa-sdk-go/NexaCLI)

---

### 🐍 Python SDK

```bash
pip install nexaai
```

```python
from nexaai import LLM, GenerationConfig, ModelConfig, LlmChatMessage

llm = LLM.from_(model="NexaAI/Qwen3-0.6B-GGUF", config=ModelConfig())

conversation = [
    LlmChatMessage(role="user", content="Hello, tell me a joke")
]
prompt = llm.apply_chat_template(conversation)
for token in llm.generate_stream(prompt, GenerationConfig(max_tokens=100)):
    print(token, end="", flush=True)
```

- **Models:** LLM, Multimodal, ASR, OCR, Rerank, Object Detection, Image Generation, Embedding
- **Formats:** GGUF, NEXA
- 📖 [Python SDK Docs](https://docs.nexa.ai/en/nexa-sdk-python/quickstart)

---

### 🤖 Android SDK

Add to your `app/AndroidManifest.xml`

```xml
<application android:extractNativeLibs="true">
```

Add to your `build.gradle.kts`:

```kotlin
dependencies {
    implementation("ai.nexa:core:0.0.19")
}
```

```kotlin
// Initialize SDK
NexaSdk.getInstance().init(this)

// Load and run model
VlmWrapper.builder()
    .vlmCreateInput(VlmCreateInput(
        model_name = "omni-neural",
        model_path = "/data/data/your.app/files/models/OmniNeural-4B/files-1-1.nexa",
        plugin_id = "npu",
        config = ModelConfig()
    ))
    .build()
    .onSuccess { vlm ->
        vlm.generateStreamFlow("Hello!", GenerationConfig()).collect { print(it) }
    }
```

- **Requirements:** Android minSdk 27, Qualcomm Snapdragon 8 Gen 4 Chip
- **Models:** LLM, Multimodal, ASR, OCR, Rerank, Embedding
- **NPU Models:** [Supported Models](https://docs.nexa.ai/en/nexa-sdk-android/overview#supported-models)
- 📖 [Android SDK Docs](https://docs.nexa.ai/en/nexa-sdk-android/quickstart)

---

### 🐳 Linux Docker

```bash
docker pull nexa4ai/nexasdk:latest

export NEXA_TOKEN="your_token_here"
docker run --rm -it --privileged \
  -e NEXA_TOKEN \
  nexa4ai/nexasdk:latest infer NexaAI/Granite-4.0-h-350M-NPU
```

- **Requirements:** Qualcomm Dragonwing IQ9, ARM64 systems
- **Models:** LLM, VLM, ASR, CV, Rerank, Embedding
- **NPU Models:** [Supported Models](https://docs.nexa.ai/en/nexa-sdk-docker/overview#supported-models)
- 📖 [Linux Docker Docs](https://docs.nexa.ai/en/nexa-sdk-docker/quickstart)

---

## ⚙️ Features & Comparisons

<div align="center">

| Features                                 | **NexaSDK**                                                | **Ollama** | **llama.cpp** | **LM Studio** |
| ---------------------------------------- | ---------------------------------------------------------- | ---------- | ------------- | ------------- |
| NPU support                              | ✅ NPU-first                                               | ❌         | ❌            | ❌            |
| Android SDK support                  | ✅ NPU/GPU/CPU support                                     | ⚠️         | ⚠️            | ❌            |
| Linux support (Docker image)             | ✅                                                         | ✅         | ✅            | ❌            |
| Day-0 model support  | ✅                                                         | ❌         | ⚠️            | ❌            |
| Full multimodality support               | ✅ Image, Audio, Text, Embedding, Rerank, ASR, TTS         | ⚠️         | ⚠️            | ⚠️            |
| Cross-platform support                   | ✅ Desktop, Mobile (Android), Automotive, IoT (Linux) | ⚠️         | ⚠️            | ⚠️            |
| One line of code to run                  | ✅                                                         | ✅         | ⚠️            | ✅            |
| OpenAI-compatible API + Function calling | ✅                                                         | ✅         | ✅            | ✅            |

<p align="center" style="margin-top:14px">
  <i>
      <b>Legend:</b>
      <span title="Full support">✅ Supported</span> &nbsp; | &nbsp;
      <span title="Partial or limited support">⚠️ Partial or limited support </span> &nbsp; | &nbsp;
      <span title="Not Supported">❌ No</span>
  </i>
</p>
</div>

## 🙏 Acknowledgements

We would like to thank the following projects:

- [ggml](https://github.com/ggml-org/ggml)
- [mlx-lm](https://github.com/ml-explore/mlx-lm)
- [mlx-vlm](https://github.com/Blaizzy/mlx-vlm)
- [mlx-audio](https://github.com/Blaizzy/mlx-audio)

## 📄 License

NexaSDK uses a dual licensing model:

### CPU/GPU Components

Licensed under [Apache License 2.0](LICENSE).

### NPU Components

- **Personal Use**: Free license key available from [Nexa AI Model Hub](https://sdk.nexa.ai/model). Each key activates 1 device for NPU usage.
- **Commercial Use**: Contact [hello@nexa.ai](mailto:hello@nexa.ai) for licensing.

## 🤝 Contact & Community Support

Want more model support, backend support, device support or other features? We'd love to hear from you!

Feel free to [submit an issue](https://github.com/NexaAI/nexa-sdk/issues) on our GitHub repository with your requests, suggestions, or feedback. Your input helps us prioritize what to build next.
