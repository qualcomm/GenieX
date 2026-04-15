def _sdk_local_bundle_impl(ctx):
    root_symlinks = {}
    strip_prefix = ctx.attr.strip_prefix

    for f in ctx.files.data:
        if not f.short_path.startswith(strip_prefix):
            fail("file %s does not start with strip_prefix %s" % (f.short_path, strip_prefix))
        root_symlinks['_main/' + f.short_path[len(strip_prefix):]] = f

    runfiles = ctx.runfiles(root_symlinks = root_symlinks)

    return DefaultInfo(
        files = depset(),
        default_runfiles = runfiles,
        data_runfiles = runfiles,
    )

sdk_local_bundle = rule(
    implementation = _sdk_local_bundle_impl,
    attrs = {
        "data": attr.label_list(allow_files = True),
        "strip_prefix": attr.string(mandatory = True),
    },
)
