# Build

## SDK

Windows ARM64:

```powershell
$env:BAZEL_VS='C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools'
$env:BAZEL_VC='C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC'
bazelisk build //sdk/src:libgeniex
```

Output:

```text
bazel-out/arm64_windows-fastbuild/bin/sdk/src/libgeniex.dll
```

Test build:

```powershell
bazelisk build //sdk/tests/src:geniex_test_version
```

## Notes

- SDK Bazel target name: `//sdk/src:libgeniex`
- CMake is not required for the Bazel SDK build
- Plugin DLL Bazel targets are not wired yet