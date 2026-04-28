BRIDGE_VERSION = "v1.0.45-rc1"

def _sdk_s3_repo_impl(ctx):
    os_name_map = {
        "linux": "linux",
        "windows 11": "windows",
    }
    os_name = os_name_map.get(ctx.os.name)
    if os_name == None:
        fail("unsupported host os for sdk s3 repo: %s" % ctx.os.name)

    arch_map = {
        "amd64": "x86_64",
        "aarch64": "arm64",
    }
    os_arch = arch_map.get(ctx.os.arch)
    if os_arch == None:
        fail("unsupported host arch for sdk s3 repo: %s" % ctx.os.arch)

    bridge_version = ctx.attr.bridge_version
    base_url = "https://nexa-model-hub-bucket.s3.us-west-1.amazonaws.com/public/nexasdk"
    url = "%s/%s/%s_%s/nexasdk-bridge.zip" % (base_url, bridge_version, os_name, os_arch)
    ctx.download_and_extract(url = url, output = "build")

    ctx.file("BUILD.bazel", """
load("@rules_cc//cc:defs.bzl", "cc_import", "cc_library")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "sdk",
    hdrs = ["build/geniex.h"],
    includes = ["build"],
    srcs = select({
        "@platforms//os:windows": ["build/nexa_bridge.dll"],
        "@platforms//os:linux": ["build/libnexa_bridge.so"],
    }),
    data = glob(["build/**/*"]),
)
""")


s3_sdk_download = repository_rule(
    implementation = _sdk_s3_repo_impl,
    attrs = {
        "bridge_version": attr.string(default = BRIDGE_VERSION),
    },
)
