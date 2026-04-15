# Build CLI

This repository uses Bazel and Bazelisk for SDK and plugin builds.

Install Bazelisk:

- Windows: `winget install -e --id Bazel.Bazelisk`
- Linux: install `bazelisk` from your package manager

Then just build and run cli with `bazelisk run //cli -- infer Qwen/Qwen3-0.6B-GGUF`, all dependencies will be automatically downloaded and built by Bazel.

There are also some optional flags for `bazelisk run`:
- `--//flags:sdk_type=s3` default behavior, download prebuilt sdk binaries from s3, which is the recommended way to get the sdk for most users
- `--//flags:sdk_type=local` to force local build of sdk instead of using prebuilt binaries, you should manually build the sdk first, see [Build SDK](#build-sdk) section below
- `--//flags:sdk_type=bazel` WIP


# Build SDK
