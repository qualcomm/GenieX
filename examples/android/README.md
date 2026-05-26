# DO NOT COPY
```bash
# ===== GENIEX INTERNAL BEGIN =====
# TODO: put all credentials here
# ===== GENIEX INTERNAL END =====
```

# Geniex Android SDK Demo App

## Overview

The Geniex AI Android SDK enables on-device AI inference for Android applications with NPU acceleration. Run Large Language Models (LLMs), Vision-Language Models (VLMs), Embeddings, Speech Recognition (ASR), Reranking, and Computer Vision models on Android devices with support for NPU, GPU, and CPU inference. Follow [Android SDK Doc](https://docs.geniex.ai/geniex-sdk-android/overview) for more details. This folder contains the demo app for the Android SDK.

## Device Compatibility

### Supported Hardware

- **NPU**: Qualcomm Snapdragon 8 Elite, 8 Elite Gen 5
- **GPU**: Qualcomm Adreno GPU
- **CPU**: ARM64-v8a
- **RAM**: 4GB+ recommended
- **Storage**: 100MB - 4GB (varies by model)

### Minimum Requirements

- Android API Level 27+ (Android 8.1 Oreo)
- **Architecture**: ARM64-v8a
- **Android SDK Version**: 27+

## Build and Run

1. Clone the repository

```bash
git clone https://github.com/GeniexAI/geniex-sdk/
```

2. Open this folder `bindings/android` in Android Studio

3. Follow the instructions in docs, download model, and put it in App's assets folder.

4. Build and run the app
