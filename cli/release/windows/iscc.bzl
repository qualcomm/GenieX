"""Repository rule to fetch InnoSetup and expose iscc.exe."""

def _innosetup_impl(ctx):
    ctx.download(
        url = ctx.attr.url,
        output = "innosetup_installer.exe",
        sha256 = ctx.attr.sha256,
    )

    result = ctx.execute([
        str(ctx.path("innosetup_installer.exe")),
        "/VERYSILENT",
        "/SUPPRESSMSGBOXES",
        "/NORESTART",
        "/SP-",
        "/DIR=" + str(ctx.path(".")),
    ], timeout = 120)

    if result.return_code != 0:
        fail("InnoSetup install failed: " + result.stderr)

    ctx.delete(ctx.path("innosetup_installer.exe"))

    ctx.file("BUILD.bazel", """
filegroup(
    name = "iscc",
    srcs = ["ISCC.exe"],
    visibility = ["//visibility:public"],
)
""")

innosetup_repository = repository_rule(
    implementation = _innosetup_impl,
    attrs = {
        "url": attr.string(default = "https://github.com/jrsoftware/issrc/releases/download/is-6_7_1/innosetup-6.7.1.exe"),
        "sha256": attr.string(default = ""),
    },
)
