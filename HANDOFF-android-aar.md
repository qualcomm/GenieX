# Handoff: Build Android AAR locally to unblock demo testing

**Audience:** an agent or human working on a Linux x64 workstation with Docker.

**Goal:** produce an unpublished `geniex-android-0.2.4-SNAPSHOT.aar` (or any local
artifact id) containing the `@JvmStatic` fix for `ModelType.fromValue`, then make
it available to the demo APK at `examples/android/` without waiting for the
v0.2.4 Maven Central publish.

The local Windows host (where this work started) cannot do this:
1. The Snapdragon Android toolchain image is `linux/amd64` only.
2. Docker Desktop on this Windows host has the Linux engine disabled
   (`docker version --format '{{.Server.Os}}/{{.Server.Arch}}'` errors out
   with `dockerDesktopLinuxEngine` not running).

## Repo state

- Branch: `perry/refactor/android-app-align`
- HEAD: `568fdc1b49dc67e78a65c9a6d2621ad1c0afea59`
- Tag `v0.2.4` is pushed at this same SHA — release CI is running, but the
  user wants a local AAR to test before Maven Central propagation completes.
- The `@JvmStatic` fix for the JNI crash is in commit `4ce6648`
  (`bindings/android/app/src/main/java/com/geniex/sdk/bean/ModelType.kt`).

## Why a fresh SDK package is required

`bindings/android/app/src/main/cpp/model_manager_jni.cpp` references symbols
that the **stale** `sdk/pkg-geniex/` (built 2026-05-22) does not export:

- `geniex_ModelListDetailedOutput` (current name) vs old `geniex_ModelListOutput`
- `geniex_model_list_detailed` vs old `geniex_model_list`
- `geniex_ModelPullInput.model_type` field
- `geniex_model_last_error_message`
- `geniex_ModelPaths.plugin_id` / `model_type`

The source-of-truth header at `sdk/model-manager/include/geniex_model.h`
already has these. So the SDK Android package needs to be rebuilt before the
AAR will compile.

## Steps

### 1. Clone and check out

```bash
git clone github-work:qualcomm/nexa-sdk.git
cd geniex
git fetch origin perry/refactor/android-app-align
git checkout perry/refactor/android-app-align
git submodule update --init --recursive
```

Verify HEAD is `568fdc1` (or v0.2.4).

### 2. Rebuild the Android SDK package

Per [`notes/build.md` § Android (cross-compile from Linux)](notes/build.md):

```bash
docker run --rm -u $(id -u):$(id -g) \
    --volume $(pwd):/workspace \
    --workdir /workspace/sdk \
    -e CCACHE_DIR=/workspace/.ccache \
    --platform linux/amd64 \
    docker.io/qualcomm/geniex-toolchain-android:v0.0.1 \
    bash -c 'cmake --preset arm64-android-snapdragon-debug -B build-android . \
      && cmake --build build-android -j \
      && cmake --install build-android --prefix pkg-geniex'
```

Expected output: `sdk/pkg-geniex/` populated with arm64 Android `.so` files
(check `file sdk/pkg-geniex/lib/libgeniex.so` says `ARM aarch64 ... for Android`)
and the new headers under `sdk/pkg-geniex/include/` containing
`geniex_ModelListDetailedOutput` etc.

### 3. Build the AAR

```bash
cd bindings/android
./gradlew --no-daemon assembleRelease
```

Output: `bindings/android/app/build/outputs/aar/app-release.aar`.

Verify the fix is in there:
```bash
unzip -p app/build/outputs/aar/app-release.aar classes.jar > /tmp/classes.jar
javap -p -classpath /tmp/classes.jar com.geniex.sdk.bean.ModelType | grep -i fromValue
# Expect a `public static ModelType fromValue(int)` (no Companion-only form).
```

### 4. Hand the AAR back to the Windows host

Easiest: push it to a branch / artifact / S3 path the user can `curl`. Or
attach to a draft GitHub release. Then on the Windows host the user does:

```bash
# In a local Maven repo:
LOCAL_REPO="$LOCALAPPDATA/Temp/local-maven/com/qualcomm/qti/geniex-android/0.2.4-local"
mkdir -p "$LOCAL_REPO"
cp app-release.aar "$LOCAL_REPO/geniex-android-0.2.4-local.aar"

# Minimal POM:
cat > "$LOCAL_REPO/geniex-android-0.2.4-local.pom" <<'POM'
<?xml version="1.0"?>
<project xmlns="http://maven.apache.org/POM/4.0.0">
  <modelVersion>4.0.0</modelVersion>
  <groupId>com.qualcomm.qti</groupId>
  <artifactId>geniex-android</artifactId>
  <version>0.2.4-local</version>
  <packaging>aar</packaging>
</project>
POM
```

And the demo's Gradle init script (`examples/android/local-aar-init.gradle`,
gitignored — already exists, see commit `4120f37`):

```groovy
beforeSettings { settings ->
    settings.dependencyResolutionManagement {
        repositories {
            maven {
                url = uri('file:///C:/Users/zhic/AppData/Local/Temp/local-maven')
                content { includeModule('com.qualcomm.qti', 'geniex-android') }
            }
        }
    }
}
```

Bump the demo's pin from `0.2.3` to `0.2.4-local` in:
- `examples/android/build.gradle:59`

Then:
```powershell
cd examples/android
./gradlew assembleDebug --init-script local-aar-init.gradle
adb install -r build/outputs/apk/debug/app-debug.apk
```

## Notes / gotchas

- The Snapdragon Android toolchain image is `linux/amd64`-only.
  `--platform linux/amd64` is required even on x86_64 hosts so buildx doesn't
  pick a default; on arm64 hosts it will use qemu.
- `--user $(id -u):$(id -g)` keeps build outputs owned by the host user, not
  root. Skipping it leaves files only `sudo rm -rf` can clean.
- ccache is mounted at `.ccache/` to make incremental rebuilds reasonable;
  add it to gitignore on the working machine if it isn't already.
- This handoff exists because the v0.2.4 Maven Central publish is the
  permanent solution — once it lands, the gradle pin can move to plain
  `0.2.4` and `local-aar-init.gradle` is no longer needed. Don't commit
  any of the local-Maven plumbing.
