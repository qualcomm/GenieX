load("@rules_cc//cc/common:cc_info.bzl", "CcInfo")

def _sdk_local_bundle_impl(ctx):
    sdk_default_info = ctx.attr.sdk_lib[DefaultInfo]

    root_symlinks = {}
    strip_prefix = ctx.attr.strip_prefix or ""
    prefix = ctx.attr.runfiles_prefix or ""
    if prefix:
        prefix = prefix.strip('/')

    for file in ctx.files.data:
        short_path = file.short_path
        if strip_prefix:
            if not short_path.startswith(strip_prefix):
                fail("expected %s to start with %s" % (short_path, strip_prefix))
            rel = short_path[len(strip_prefix):]
        else:
            rel = short_path
        if rel.startswith("/"):
            rel = rel[1:]
        if prefix:
            key = prefix + "/" + rel if rel else prefix
        else:
            key = rel
        root_symlinks[key] = file

    runfiles = ctx.runfiles(root_symlinks = root_symlinks)

    return [
        ctx.attr.sdk_lib[CcInfo],
        DefaultInfo(
            files = sdk_default_info.files,
            default_runfiles = sdk_default_info.default_runfiles.merge(runfiles),
            data_runfiles = sdk_default_info.data_runfiles.merge(runfiles),
        ),
    ]

sdk_local_bundle = rule(
    implementation = _sdk_local_bundle_impl,
    attrs = {
        "sdk_lib": attr.label(
            mandatory = True,
            providers = [CcInfo, DefaultInfo],
        ),
        "data": attr.label_list(
            allow_files = True,
        ),
        "strip_prefix": attr.string(
            mandatory = True,
        ),
        "runfiles_prefix": attr.string(
            default = "_main",
        ),
    },
)
