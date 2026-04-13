load(
    "@rules_license//rules:providers.bzl",
    "LicenseInfo",
    "LicenseKindInfo",
)


def license_rule_impl(ctx):
    return [
        LicenseInfo(
            copyright_notice = ctx.attr.copyright_notice,
            label = ctx.label,
            license_kinds = tuple([dep[LicenseKindInfo] for dep in ctx.attr.license_kinds]),
            license_text = ctx.file.license_text,
            package_name = ctx.attr.package_name or ctx.label.package,
            package_url = ctx.attr.package_url,
            package_version = ctx.attr.package_version,
        ),
    ]