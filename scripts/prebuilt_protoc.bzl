"""Downloads a prebuilt protoc matching the exec platform and wraps it
as a proto_toolchain.

Protobuf 33.4's own prebuilt_protoc extension does not publish a
Windows ARM64 binary. Windows 11 ARM64 can execute the win64 (x86_64)
protoc.exe under x64 emulation, so we reuse that archive here.
"""

_VERSION = "33.4"

_ARTIFACTS = {
    # (os, cpu): (filename, sha256)
    ("windows", "arm64"): (
        "protoc-33.4-win64.zip",
        "0b31be019b9fe45a388e93bf1f16d70afdf9ce5caf958ea47892fbc868b5a1b3",
    ),
}

_BUILD = """\
load("@com_google_protobuf//bazel/toolchains:proto_toolchain.bzl", "proto_toolchain")

package(default_visibility = ["//visibility:public"])

# `proto_toolchain` emits two targets: `:prebuilt` (the impl) and
# `:prebuilt_toolchain` (the registered toolchain). Register the latter
# from MODULE.bazel.
proto_toolchain(
    name = "prebuilt",
    exec_compatible_with = {exec_constraints},
    proto_compiler = "{protoc_label}",
)
"""

def _impl(rctx):
    key = (rctx.attr.os, rctx.attr.cpu)
    if key not in _ARTIFACTS:
        fail("no prebuilt protoc override for {}".format(key))
    filename, sha256 = _ARTIFACTS[key]
    rctx.download_and_extract(
        url = "https://github.com/protocolbuffers/protobuf/releases/download/v{}/{}".format(
            _VERSION,
            filename,
        ),
        sha256 = sha256,
    )
    rctx.file("BUILD.bazel", _BUILD.format(
        protoc_label = "bin/protoc.exe",
        exec_constraints = str([
            "@platforms//os:{}".format(rctx.attr.os),
            "@platforms//cpu:{}".format(rctx.attr.cpu),
        ]),
    ))

prebuilt_protoc = repository_rule(
    implementation = _impl,
    attrs = {
        "os": attr.string(mandatory = True),
        "cpu": attr.string(mandatory = True),
    },
)
