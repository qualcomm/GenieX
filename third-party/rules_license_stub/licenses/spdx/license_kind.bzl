load("@rules_license//rules:providers.bzl", "LicenseKindInfo")


def _license_kind_impl(ctx):
    return [
        LicenseKindInfo(
            conditions = ctx.attr.conditions,
            label = ctx.label,
            long_name = ctx.attr.long_name,
            name = ctx.label.name,
        ),
    ]


_license_kind_rule = rule(
    implementation = _license_kind_impl,
    attrs = {
        "conditions": attr.string_list(),
        "long_name": attr.string(),
    },
)


def license_kind(**kwargs):
    if "visibility" not in kwargs:
        kwargs["visibility"] = ["//visibility:public"]
    if "applicable_licenses" not in kwargs:
        kwargs["applicable_licenses"] = []
    _license_kind_rule(**kwargs)