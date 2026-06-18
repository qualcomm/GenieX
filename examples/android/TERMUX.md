# Building on Termux (aarch64)

Termux provides a full Linux environment on Android — you can clone, edit,
compile, and sideload APKs entirely on-device without a desktop machine.
This guide covers the extra steps needed to build the GenieX Android demo on
aarch64 Termux, where the standard Android build toolchain has pitfalls.

## Why build on Termux?

- **No desktop required** — develop and build directly on your phone or tablet.
- **Faster iteration** — edit a Kotlin file, `gradlew assembleDebug`, install
  the APK, run it. All on the same device.
- **Keep everything on-device** — models, SDK, and source all live under one
  filesystem; no adb push/pull dance.

## Prerequisites

### Packages

```
pkg install openjdk-21 gradle aapt2
```

You also need `git` (usually pre-installed) and `python3` for scripting.

### Android SDK

Install the Android SDK at `$HOME/android-sdk`. The easiest path is
`cmdline-tools` + `sdkmanager`:

```bash
mkdir -p ~/android-sdk/cmdline-tools
# Download latest command-line tools from developer.android.com
# Extract into ~/android-sdk/cmdline-tools/latest/
export ANDROID_HOME=$HOME/android-sdk
~/android-sdk/cmdline-tools/latest/bin/sdkmanager --install \
  "platforms;android-34" \
  "platforms;android-35" \
  "platforms;android-36" \
  "build-tools;34.0.0" \
  "build-tools;34.0.4" \
  "build-tools;35.0.1" \
  "build-tools;36.0.0"
```

Exact contents of `$HOME/android-sdk/build-tools/` after setup:

```
34.0.0/  34.0.4/  35.0.0/  35.0.1/  36.0.0/
```

And `$HOME/android-sdk/platforms/`:

```
android-33/  android-34/  android-35/  android-36/
```

## Workarounds

Three things must be changed for the build to succeed on aarch64 Termux.

### 1. local.properties

Create `local.properties` pointing at the SDK:

```properties
sdk.dir=/data/data/com.termux/files/home/android-sdk
```

> **Never commit this file.** It is already in `.gitignore`.

### 2. aapt2 override (gradle.properties)

The aarch64 aapt2 binary shipped with build-tools older than 34.x **cannot
load platform JARs >= 35**. AGP also bundles an x86_64 aapt2 that won't run
on aarch64. The fix is to tell Gradle to use the build-tools 34.0.4 aapt2,
which is aarch64-native:

```properties
android.aapt2FromMavenOverride=/data/data/com.termux/files/home/android-sdk/build-tools/34.0.4/aapt2
```

The system `aapt2` from `pkg install aapt2` also works and is sometimes
newer — if your build fails during resource linking, try swapping.

### 3. compileSdk must be 34

Because of the aapt2 limitation above, `compileSdk` and `targetSdk` in
`build.gradle` must be 34:

```groovy
ANDROID_COMPILE_API = 34
ANDROID_TARGET_API  = 34
```

> With compileSdk 35 or higher, processDebugResources fails with:
> `AAPT: error: failed to load include path .../android-35/android.jar`

### 4. NDK version

The `build.gradle` declares `ANDROID_NDK_VERSION = '29.0.14206865'`, but
the example has **no native code** — AGP skips the NDK entirely. If you
later add C/C++ code, install the matching NDK:

```bash
sdkmanager --install "ndk;29.0.14206865"
```

## Build

```bash
cd examples/android
ANDROID_HOME=$HOME/android-sdk ./gradlew assembleDebug
```

First build downloads Gradle 9.1.0 and all dependencies — ~2-3 minutes on
a warm cache, longer on first run.

APK output: `build/outputs/apk/debug/app-debug.apk`

### Incremental rebuilds

```bash
./gradlew assembleDebug
```

### Clean rebuild (if you switch aapt2 or compileSdk)

```bash
rm -rf ~/.gradle/caches/*/transforms/*aapt*
./gradlew assembleDebug --no-build-cache
```

## Troubleshooting

| Symptom | Fix |
|---|---|
| `AAPT: error: failed to load include path .../android-35/android.jar` | Set compileSdk to 34 in `build.gradle` |
| `java.lang.UnsatisfiedLinkError` for aapt2 | Check `android.aapt2FromMavenOverride` points to the aarch64 aapt2 |
| Gradle can't find SDK | Create `local.properties` with correct `sdk.dir` |
| NDK not found | Remove or update `ANDROID_NDK_VERSION` in `build.gradle` (no native code = not needed) |
| Out of memory during dex | Reduce `-Xmx` in `gradle.properties` or close other apps |

## Installing the APK

```bash
# If you have the APK on the same device:
termux-open build/outputs/apk/debug/app-debug.apk

# Or from adb (if you have a second device / PC):
adb install build/outputs/apk/debug/app-debug.apk
```

## Reference

- [browseragent TERMUX.md](https://github.com/sigsegv0x0b/browseragent) —
  similar aarch64 build setup, originally solved many of these issues.
- [Android SDK CLI tools](https://developer.android.com/studio#command-line-tools-only) —
  official download page.
