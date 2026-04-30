"""Merge tars with last-writer-wins dedup on path collisions.

Extracts each input tar in order into a scratch directory (so later inputs
overwrite earlier ones on path collisions), with usrmerge rewrites applied,
then repacks into a single tar.
"""

_TAR_TOOLCHAIN_TYPE = "@tar.bzl//tar/toolchain:type"

_USRMERGE = " ".join([
    "-s ',^\\./bin/,./usr/bin/,'",
    "-s ',^\\./sbin/,./usr/sbin/,'",
    "-s ',^\\./lib/,./usr/lib/,'",
    "-s ',^\\./bin$,./usr/bin,'",
    "-s ',^\\./sbin$,./usr/sbin,'",
    "-s ',^\\./lib$,./usr/lib,'",
])

def _dedup_tar_impl(ctx):
    bsdtar = ctx.toolchains[_TAR_TOOLCHAIN_TYPE]
    output = ctx.actions.declare_file(ctx.attr.name + ".tar")
    tar_bin = bsdtar.tarinfo.binary.path

    command = """
set -euo pipefail
scratch=$(mktemp -d)
trap 'rm -rf "$scratch"' EXIT
for f in "$@"; do
    "{tar}" -xpf "$f" -C "$scratch" {usrmerge}
done
"{tar}" -cf "{out}" -C "$scratch" .
""".format(tar = tar_bin, out = output.path, usrmerge = _USRMERGE)

    ctx.actions.run_shell(
        outputs = [output],
        inputs = ctx.files.tars,
        tools = [bsdtar.default.files],
        arguments = [f.path for f in ctx.files.tars],
        command = command,
        mnemonic = "DedupTar",
        progress_message = "Dedup-merging tars into %{output}",
    )

    return [DefaultInfo(files = depset([output]))]

dedup_tar = rule(
    implementation = _dedup_tar_impl,
    attrs = {
        "tars": attr.label_list(
            allow_files = True,
            mandatory = True,
        ),
    },
    toolchains = [_TAR_TOOLCHAIN_TYPE],
)
