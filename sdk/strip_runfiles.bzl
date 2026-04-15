"""
Utility rule: strip_runfiles

Usage:
  load("//sdk:strip_runfiles.bzl", "strip_runfiles")

  strip_runfiles(
      name = "my_runfiles",
      srcs = glob(["build/**/*"]),
      strip_prefix = "sdk/build/",
      runfiles_prefix = "_main",  # optional
  )

This maps files under `sdk/build/...` into runfiles with the `sdk/build/` prefix removed.
"""

def _impl(ctx):
    strip = ctx.attr.strip_prefix or ""
    prefix = ctx.attr.runfiles_prefix or ""
    if prefix:
        prefix = prefix.strip('/')
    files = ctx.files.srcs

    root_symlinks = {}
    for f in files:
        sp = f.short_path
        if strip:
            if not sp.startswith(strip):
                fail("file %s does not start with strip_prefix %s" % (sp, strip))
            rel = sp[len(strip):]
        else:
            rel = sp
        # avoid leading slash
        if rel.startswith("/"):
            rel = rel[1:]
        if prefix:
            key = prefix + "/" + rel if rel else prefix
        else:
            key = rel
        root_symlinks[key] = f

    runfiles = ctx.runfiles(root_symlinks = root_symlinks)

    return DefaultInfo(
        files = depset(files),
        default_runfiles = runfiles,
        data_runfiles = runfiles,
    )

strip_runfiles = rule(
    implementation = _impl,
    attrs = {
        "srcs": attr.label_list(allow_files = True),
        "strip_prefix": attr.string(),
        "runfiles_prefix": attr.string(),
    },
)
