"""Emit an InnoSetup [Files] snippet from a pkg_filegroup."""

load("@rules_pkg//pkg:providers.bzl", "PackageFilegroupInfo")

def _impl(ctx):
    out = ctx.actions.declare_file(ctx.attr.name + ".iss")
    files = []
    lines = []
    for pf, _ in ctx.attr.src[PackageFilegroupInfo].pkg_files:
        for dest, f in pf.dest_src_map.items():
            files.append(f)
            parts = dest.replace("/", "\\").rsplit("\\", 1)
            sub = "\\" + parts[0] if len(parts) == 2 else ""
            lines.append(
                'Source: "{#SrcRoot}\\%s"; DestDir: "{app}%s"; Flags: ignoreversion' %
                (f.path.replace("/", "\\"), sub),
            )
    ctx.actions.write(out, "\n".join(lines) + "\n")
    return [DefaultInfo(files = depset([out] + files))]

iss_manifest = rule(
    implementation = _impl,
    attrs = {
        "src": attr.label(mandatory = True, providers = [PackageFilegroupInfo]),
    },
)
