# GenieX Android SDK Demo App

## Overview

The GenieX AI Android SDK enables on-device AI inference for Android applications with NPU acceleration. Run Large Language Models (LLMs), Vision-Language Models (VLMs) on Android devices with support for NPU, GPU, and CPU inference. This folder contains the demo app for the Android SDK.

## Pick Your Path

- **Just want to try the demo?** Download the pre-built APK from the [Android install guide](https://refactored-happiness-4qyl9vn.pages.github.io/en/run/android/install/).
- **Want to build the demo app locally?** Follow the [Build and Run](#build-and-run) steps below.
- **Want to build your own Android app with the GenieX Java/Kotlin binding?** See the [Android quickstart](https://refactored-happiness-4qyl9vn.pages.github.io/en/run/android/quickstart/) for dependency setup and the [API reference](https://refactored-happiness-4qyl9vn.pages.github.io/en/run/android/api-reference/) for usage.

## Device Compatibility

### Supported Hardware

- **NPU**: Qualcomm Snapdragon 8 Elite, 8 Elite Gen 5
- **GPU**: Qualcomm Adreno GPU
- **CPU**: ARM64-v8a
- **RAM**: 8GB+ recommended
- **Storage**: 100MB - 10GB (varies by model)

### Minimum Requirements

- Android API Level 27+ (Android 8.1 Oreo)
- **Architecture**: ARM64-v8a
- **Android SDK Version**: 27+

## Build and Run

1. Clone the sdk project root repository

```bash
git clone --recursive git@github.com:qcom-ai-hub/geniex.git
```

2. Open this folder `examples/android` in Android Studio

3. Connect to your device, sync gradle, then build and run the app
