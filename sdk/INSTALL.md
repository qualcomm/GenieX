# Geniex SDK

## Build & Install

```bash
cd sdk
cmake -S . -B build
cmake --build build -j
cmake --install build --prefix pkg-geniex
```

## Installed Layout

```
pkg-geniex/
├── include/
│   └── ml.h
├── lib/
│   ├── libgeniex.so
│   ├── llama_cpp/
│   │   ├── libgeniex_plugin.so
│   │   ├── libllama.so.*
│   │   ├── libggml*.so*
│   │   ├── libmtmd.so.*
│   │   └── libcommon.a
│   └── qairt/
│       ├── libgeniex_plugin.so
│       ├── libgeniex_core.so
│       └── ...
└── bin/
    └── geniex_test_*
```

## Build Options

```bash
cmake -S . -B build \
  -DGENIEX_PLUGIN_LLAMA_CPP=ON \
  -DGENIEX_PLUGIN_QAIRT=OFF \
  -DGENIEX_DL=ON \
  -DGENIEX_DEBUG=ON
```

**Note**:

- Python bindings are not supported for cross-compilation yet
- The build uses cross-compilation toolchain (`cmake/arm64-linux-gnu.cmake`) for native ARM64 builds

---

## Android

### Prerequisites

- **Android NDK**: Required for building native libraries
- **Hexagon SDK** (Optional): The new llama.cpp `ggml-hexagon` backend.

### Build android bridge library

#### Linux

Set Android NDK path:

```bash
# Linux
# export ANDROID_NDK_ROOT=~/android-ndk-r27d
export ANDROID_NDK_ROOT=<path_to_android_ndk>
```

Next, build the bridge library:

**Without Hexagon SDK:**

```bash
# build for Android app (without Hexagon DSP support - recommended)
bash scripts/build_android.sh --no-hexagon

# build for unit testing
bash scripts/build_android.sh --no-hexagon --mode test
```

**With Hexagon SDK:**

Due to docker resource limits, you might need to change `-j$(nproc)` to `-j1` in `scripts/build_android.sh` to avoid running out of memory.

```bash
# build docker and enter the container with root privileges (-u 0:0)
docker run -it --rm -u 0:0 --volume $(pwd):/workspace --platform linux/amd64 ghcr.io/snapdragon-toolchain/arm64-android:v0.3 /bin/bash

# within the container, install build tools
apt-get update && apt-get install -y make cmake build-essential

# go to workspace and clean build directory to avoid conflicts
cd /workspace
rm -rf build-android

# build for Android app (with Hexagon DSP support)
bash scripts/build_android.sh

# build for unit testing (requires Hexagon SDK)
bash scripts/build_android.sh --mode test
```

#### Windows

Set Android NDK path in PowerShell:

```powershell
# Set NDK path (from Android Studio SDK location)
$env:ANDROID_NDK_ROOT = "C:\Users\<username>\AppData\Local\Android\Sdk\ndk\<ndk_version>"
```

Next, build the bridge library using PowerShell:

**Without Hexagon SDK:**

```powershell
# build for Android app (without Hexagon DSP support - recommended)
.\scripts\build_android.ps1 -NoHexagon

# build for unit testing
.\scripts\build_android.ps1 -NoHexagon -Mode test

# build with debug symbols
.\scripts\build_android.ps1 -NoHexagon -Debug
```

**Run Android Tests (Windows):**

```powershell
# run tests on connected Android device
.\scripts\run_android_tests.ps1

# specify build mode
.\scripts\run_android_tests.ps1 -Mode test
```

**PowerShell Script Options:**

| Option            | Description                                            | Default       |
| ----------------- | ------------------------------------------------------ | ------------- |
| `-Abi <abi>`      | Target ABI (arm64-v8a)                                  | arm64-v8a     |
| `-BuildDir <dir>` | Build output directory                                 | build-android |
| `-Debug`          | Build debug version                                    | Release       |
| `-Mode <mode>`    | Build mode: `app` (static linking) or `test` (dynamic) | app           |
| `-NoHexagon`      | Disable Hexagon DSP support                            | -             |
| `-Help`           | Show help message                                      | -             |

**Note**: Hexagon SDK builds on Windows require WSL or Docker with Linux environment.

### Build AAR File

#### Linux

```bash
# Linux
# export ANDROID_HOME=~/Library/Android/sdk
export ANDROID_HOME=<path_to_android_sdk>

# export ANDROID_NDK_HOME=~/Library/Android/sdk/ndk/29.0.13846066
export ANDROID_NDK_HOME=<path_to_android_ndk>

cd bindings/android
./gradlew wrapper --gradle-version 9.1.0
./gradlew :app:assembleDebug
```

#### Windows

```powershell
# Set environment variables
$env:ANDROID_HOME = "C:\Users\<username>\AppData\Local\Android\Sdk"
$env:ANDROID_NDK_HOME = "C:\Users\<username>\AppData\Local\Android\Sdk\ndk\<ndk_version>"

cd bindings\android

# If gradle-wrapper.jar is missing, use system Gradle to generate it
gradle wrapper --gradle-version 9.1.0

# Or if Gradle is not installed globally, download the wrapper JAR manually:
# New-Item -ItemType Directory -Force -Path "gradle\wrapper"
# Invoke-WebRequest -Uri "https://raw.githubusercontent.com/gradle/gradle/v8.5.0/gradle/wrapper/gradle-wrapper.jar" -OutFile "gradle\wrapper\gradle-wrapper.jar"

# Build the AAR
.\gradlew :app:assembleDebug
```

Generated `.aar` file location: `app/build/outputs/aar/`

### Remote Debugging on Android Device

1. Open wireless debugging on Android device:
   - Go to **Settings** > **About phone** > Tap **Build number** 7 times to enable Developer Options
   - Go to **Settings** > **System** > **Developer options** > Enable **Wireless debugging**
2. Pair device with your computer:
   - In **Wireless debugging** menu, tap **Pair device with pairing code**
   - Note the IP address, port, and pairing code displayed (you will use this as `<pair_port>` in steps 3–4). Also, from the **Wireless debugging** screen, note the port shown next to your device name (this will be used as `<adb_port>` in steps 5–6).
3. Set up port forwarding from Linux server to the Android device:
   - Use ssh tunnel to forward ports: `ssh -NR 127.0.0.1:<pair_port>:<android_ip>:<pair_port> user@<linux_ip>`
4. Pair the Android device:
   - Run `adb pair 127.0.0.1:<pair_port>` and enter the pairing code
5. Set up port forwarding for adb connection:
   - Run `ssh -NR 127.0.0.1:<adb_port>:<android_ip>:<adb_port> user@<linux_ip>`
6. Connect to the Android device on Linux server:
   - Run `adb connect 127.0.0.1:<adb_port>`

### Release Android Build to Maven Central

The Android build is published to Maven Central. View releases at:

- [Maven Central Repository](https://repo1.maven.org/maven2/ai/geniex/core/)
- [MVN Repository](https://mvnrepository.com/artifact/ai.geniex/core) (indexing may take 1+ day)

**Prerequisites:**

- Android Studio with Gradle support
- Maven Central account with publishing credentials
- GPG key configured for artifact signing

**Publishing Steps:**

**1. Configure Android Studio**

Open `bindings/android` in Android Studio, then enable experimental features:

- Navigate to **Settings** > **Experimental**
- Enable all experimental Gradle options

**2. Update Release Version**

Edit `update.gradle` and modify the version, this will be the version of the AAR file.

```gradle
tmpVersion = <version>  // version of the AAR file
```

**3. Generate Artifacts**

In Android Studio Gradle panel:

- Navigate to **app** > **tasks** > **publish**
- Run the publish task
- A `repo` folder will be generated in your project directory

**4. Prepare Upload Package**

```bash
cd repo
zip -r ai.zip ai/
```

**5. Upload to Maven Central**

1. Log in to [Maven Central](https://central.sonatype.com/)
2. Upload `ai.zip` via the web interface
3. Wait for validation (several minutes)
4. Click **Publish** to release

**6. Verify Deployment**

Check artifact availability:

```gradle
// Test in your Android project
dependencies {
   implementation 'ai.geniex:core:<version>'
}
```

Verify at: `https://repo1.maven.org/maven2/ai/geniex/sdk/`

**Troubleshooting:**

- **Signing errors** - Verify GPG key configuration in `gradle.properties`
- **Upload failures** - Check Maven Central credentials
- **Validation errors** - Ensure POM, sources, and javadoc are included