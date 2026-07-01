<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="GenieX-Logo-Hor-1-White.png" />
  <source media="(prefers-color-scheme: light)" srcset="GenieX-Logo-Hor-1-Black.png" />
  <img src="GenieX-Logo-Hor-1-Black.png" width="420" alt="Qualcomm AI Hub GenieX" />
</picture>

### 在高通设备上本地运行前沿 LLM 与 VLM 的最简方式

[![Status: Developer Preview](https://img.shields.io/badge/status-developer%20preview-FF6C2C?style=flat-square)](#)
[![Docs](https://img.shields.io/badge/docs-geniex.aihub.qualcomm.com-2A2AEA?style=flat-square&logo=readthedocs&logoColor=white)](https://geniex.aihub.qualcomm.com)
[![Release](https://img.shields.io/github/v/release/qualcomm/GenieX?style=flat-square&color=2A2AEA&label=release)](https://github.com/qualcomm/GenieX/releases)
[![License: BSD-3-Clause](https://img.shields.io/badge/license-BSD--3--Clause-blue?style=flat-square)](LICENSE)
[![Slack](https://img.shields.io/badge/Slack-join%20the%20community-4A154B?style=flat-square&logo=slack&logoColor=white)](https://aihub.qualcomm.com/community/slack)

[**文档**](https://geniex.aihub.qualcomm.com) · [**快速入门**](#快速入门) · [**模型**](#模型) · [**社区与联系方式**](#-社区与联系方式)

[English](README.md) · **简体中文**

</div>

---

GenieX 是一款**面向高通设备的端侧生成式 AI 推理运行时**。你几乎可以引入任意来自 Hugging Face 的 GGUF 模型——或来自 [Qualcomm AI Hub](https://aihub.qualcomm.com/models/) 的预编译模型包——只需几行代码即可在 **Hexagon NPU、Adreno GPU 或 CPU** 上本地运行。底层是同一套 C SDK，并通过 CLI、Python、Kotlin/Java、Docker 以及一个 OpenAI 兼容服务器交互。它是 Qualcomm GENIE 的社区版本。

<div align="center">
  <img src="docs/Mintlify-image/geniex_arch_v2.png" width="820" alt="GenieX 架构：CLI、Python、Java、Docker 以及 OpenAI 兼容的 Serve 接口构建在单一的 GenieX SDK 之上，由其分发到 llama.cpp 运行时（在 CPU / GPU / Hexagon HTP 内核上运行的 GGML）或 NPU 上的 Qualcomm AI Engine Direct 运行时——覆盖 Windows、Android 与 Linux。" />
</div>

## 支持的平台

GenieX **仅在高通骁龙（Qualcomm Snapdragon）上运行**。找到你的平台，然后直接跳转到你想使用的接口。

| 平台 | 示例设备 | 跳转到快速入门 |
| --- | --- | --- |
| 🪟 **Windows ARM64** *(计算)* | 骁龙 X · X Elite | [CLI](#cli) · [Python](#python) · [本地服务器](#openai-兼容服务器) |
| 🤖 **Android** *(移动)* | 骁龙 8 至尊版 · 8 至尊版 Gen 5 | [Android SDK](#android-kotlin--java) |
| 🐧 **Linux ARM64** *(物联网)* | Dragonwing QCS9075 | [CLI](#cli) · [Docker](#docker) · [Python](#python) |


> 手边没有设备？在 [Qualcomm Device Cloud](https://qdc.qualcomm.com/) 上开启一个远程会话。

---

## 快速入门

从下方选择你的接口。每个接口都遵循相同的三个步骤——**安装（Install）**、**运行（Run）** 和 **文档（Docs）**——并同时展示两种运行时：来自 Hugging Face 的 **GGUF** 模型（`llama_cpp`）和来自 Qualcomm AI Hub 的**预编译模型包**（`qairt`，NPU）。

### CLI

![Windows ARM64](https://img.shields.io/badge/Windows%20ARM64-0078D6?style=flat-square&logo=windows&logoColor=white) ![Linux ARM64](https://img.shields.io/badge/Linux%20ARM64-FCC624?style=flat-square&logo=linux&logoColor=black)

**安装**

- **Windows ARM64** —— [下载安装程序](https://github.com/qualcomm/GenieX/releases)，运行它，然后打开一个新终端。
- **Linux ARM64** —— 一行命令，无需 `sudo`：
  ```bash
  curl -fsSL https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-geniex/install.sh | sh
  ```

**运行** —— 一行命令即可与任意模型对话（对于 VLM，拖入一张图片即可）：

```bash
# 来自 Hugging Face 的 GGUF → llama.cpp（NPU / GPU / CPU）
geniex infer google/gemma-4-E4B-it-qat-q4_0-gguf

# 来自 Qualcomm AI Hub 的预编译模型包 → Qualcomm AI Engine Direct（NPU）
geniex infer ai-hub-models/Qwen2.5-VL-7B-Instruct
```

📖 **文档** —— [安装](https://geniex.aihub.qualcomm.com/en/run/cli/install) · [快速入门](https://geniex.aihub.qualcomm.com/en/run/cli/quickstart) · [命令参考](https://geniex.aihub.qualcomm.com/en/run/cli/reference)

### Python

![Windows ARM64](https://img.shields.io/badge/Windows%20ARM64-0078D6?style=flat-square&logo=windows&logoColor=white) ![Linux ARM64](https://img.shields.io/badge/Linux%20ARM64-FCC624?style=flat-square&logo=linux&logoColor=black)

**安装**

```bash
pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple geniex
```

**运行** —— 与 Hugging Face `transformers` 保持一致（`from_pretrained()` → `.generate()`）：

```python
# 来自 Hugging Face 的 GGUF → llama.cpp
from geniex import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("unsloth/Qwen3.5-2B-GGUF", precision="Q4_0")

messages = [{"role": "user", "content": "What is 2+2?"}]
prompt = model.tokenizer.apply_chat_template(messages, add_generation_prompt=True)

for chunk in model.generate(prompt, max_new_tokens=256, stream=True):
    print(chunk, end="", flush=True)

model.close()
```

```python
# 来自 Qualcomm AI Hub 的预编译模型包 → Qualcomm AI Engine Direct（NPU）
from geniex import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("ai-hub-models/Qwen3-4B")

messages = [{"role": "user", "content": "What is 2+2?"}]
prompt = model.tokenizer.apply_chat_template(messages, add_generation_prompt=True)

for chunk in model.generate(prompt, max_new_tokens=256, stream=True):
    print(chunk, end="", flush=True)

model.close()
```

📖 **文档** —— [安装](https://geniex.aihub.qualcomm.com/en/run/python/install) · [快速入门](https://geniex.aihub.qualcomm.com/en/run/python/quickstart) · [API 参考](https://geniex.aihub.qualcomm.com/en/run/python/api-reference)

### OpenAI 兼容服务器

![Windows ARM64](https://img.shields.io/badge/Windows%20ARM64-0078D6?style=flat-square&logo=windows&logoColor=white) ![Linux ARM64](https://img.shields.io/badge/Linux%20ARM64-FCC624?style=flat-square&logo=linux&logoColor=black)

**安装** —— 随 CLI 一同提供（见[上文安装](#cli)）。

**运行** —— 拉取任意模型（GGUF 或 Qualcomm AI Hub 模型包），然后提供一个 OpenAI 兼容的 API：

```bash
geniex pull ai-hub-models/Qwen3-4B-Instruct-2507
geniex serve   # 服务于 http://127.0.0.1:18181/v1
```

```bash
curl http://127.0.0.1:18181/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "ai-hub-models/Qwen3-4B-Instruct-2507",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

将任意 OpenAI 客户端指向 `http://127.0.0.1:18181/v1` 即可——无需改动代码。

📖 **文档** —— [本地服务器指南](https://geniex.aihub.qualcomm.com/en/run/cli/local-server)

### Android（Kotlin / Java）

![Android](https://img.shields.io/badge/Android-3DDC84?style=flat-square&logo=android&logoColor=white)

**安装** —— 将 SDK 添加到应用模块的 `build.gradle.kts`：

```kotlin
dependencies {
    implementation("com.qualcomm.qti:geniex-android:0.3.1")
}
```

**运行** —— 最快的上手路径是示例应用（含聊天 UI、支持 GGUF + Qualcomm AI Hub 模型包的模型选择器、VLM 支持）：

Android 演示应用位于 [`qualcomm/ai-hub-apps`](https://github.com/qualcomm/ai-hub-apps/blob/release/geniex_chat_android/README.md)。克隆它，在 Android Studio 中打开示例应用，然后点击 **Run**。

📖 **文档** —— [安装](https://geniex.aihub.qualcomm.com/en/run/android/install) · [快速入门](https://geniex.aihub.qualcomm.com/en/run/android/quickstart) · [API 参考](https://geniex.aihub.qualcomm.com/en/run/android/api-reference)

### Docker

![Linux ARM64](https://img.shields.io/badge/Linux%20ARM64-FCC624?style=flat-square&logo=linux&logoColor=black)

**安装**

```bash
docker pull docker.io/qualcomm/geniex:latest
```

**运行** —— 容器封装了 CLI，因此 `geniex infer …` 的用法与上文完全一致。

📖 **文档** —— [Docker 指南](https://geniex.aihub.qualcomm.com/en/run/linux/install)

### C / C++ SDK

![Windows ARM64](https://img.shields.io/badge/Windows%20ARM64-0078D6?style=flat-square&logo=windows&logoColor=white) ![Linux ARM64](https://img.shields.io/badge/Linux%20ARM64-FCC624?style=flat-square&logo=linux&logoColor=black) ![Android](https://img.shields.io/badge/Android-3DDC84?style=flat-square&logo=android&logoColor=white)

**安装** —— 链接单一的 C 头文件 [`sdk/include/geniex.h`](sdk/include/geniex.h)；其他所有接口都只是它之上的一层薄封装。

📖 **文档** —— [sdk/README.md](sdk/README.md) · [notes/build.md](notes/build.md)

---

## 模型

GenieX 拥有两套运行时，让你在同一技术栈中既能获得**广泛的模型覆盖**，又能获得**骁龙上的峰值性能**。LLM 与 VLM 均受支持。

| | **llama.cpp**（`llama_cpp`） | **Qualcomm AI Engine Direct**（`qairt`） |
| --- | --- | --- |
| **模型来源** | [Hugging Face](https://huggingface.co/models?library=gguf)（任意 GGUF） | [Qualcomm AI Hub](https://aihub.qualcomm.com/models/)（预编译） |
| **格式** | GGUF | 按芯片组打包 |
| **计算单元** | NPU · GPU · CPU | 仅 NPU |
| **最适合** | 引入你自己的 GGUF | 最高的 NPU 性能 |


> 对于 llama.cpp，在提示时请选择 **`Q4_0`** 精度——它对 Hexagon NPU 的支持最佳。完整的模型列表、精度以及如何运行本地模型，请参阅[模型指南 →](https://geniex.aihub.qualcomm.com/en/models/supported)。


## 🤝 贡献

欢迎贡献！在提交 PR 之前，请阅读 **[CONTRIBUTING.md](CONTRIBUTING.md)**，了解分支命名、提交 / PR 标题格式、预提交检查，以及针对公共 SDK 头文件的 FFI 更新规则。

| | |
| --- | --- |
| 🏗️ **构建** CLI、SDK 或 Python 绑定 | [notes/build.md](notes/build.md) |
| ▶️ **运行** 并选择计算单元 / 拉取模型 | [notes/run.md](notes/run.md) |
| 🏷️ **发布** —— SemVer 标签、渠道、HTP 签名 | [notes/release.md](notes/release.md) |
| 📚 **全部开发者文档** | [docs/README.md](docs/README.md) |



## 💬 社区与联系方式

有问题、想法，或想展示你的作品？欢迎来打个招呼。

- 💬 [**Slack**](https://aihub.qualcomm.com/community/slack) —— 提问并与社区实时交流。
- 🐛 [**GitHub Issues**](https://github.com/qualcomm/GenieX/issues) —— 报告缺陷或提出功能请求。
- 🔗 [**LinkedIn**](https://www.linkedin.com/company/qualcommaihub) —— 关注 Qualcomm AI Hub 获取新闻与更新。

### 贡献者

感谢每一位共建 GenieX 的伙伴 💙

<a href="https://github.com/qualcomm/GenieX/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=qualcomm/GenieX&max=20" alt="GenieX contributors" />
</a>

---

## 📄 许可证

BSD 3-Clause —— 参阅 [LICENSE](LICENSE) 与 [NOTICE](NOTICE)。

本项目的使用还须遵守高通的[使用条款](https://www.qualcomm.com/site/terms-of-use)。
