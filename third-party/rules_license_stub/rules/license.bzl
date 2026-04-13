load("@rules_license//rules:license_impl.bzl", "license_rule_impl")
load("@rules_license//rules:providers.bzl", "LicenseKindInfo")

_require_license_text_is_a_file = False

_license = rule(
    implementation = license_rule_impl,
    attrs = {
        "license_kinds": attr.label_list(
            mandatory = False,
            cfg = "exec",
            providers = [LicenseKindInfo],
        ),
        "license_text": attr.label(allow_single_file = True),
        "package_name": attr.string(),
        "package_url": attr.string(),
        "package_version": attr.string(),
        "copyright_notice": attr.string(),
    },
)

def license(
        name,
        license_text = "LICENSE",
        license_kind = None,
        license_kinds = None,
        copyright_notice = None,
        package_name = None,
        package_url = None,
        package_version = None,
        namespace = None,
        tags = [],
        visibility = ["//visibility:public"]):
    if license_kind:
        if license_kinds:
            fail("Can not use both license_kind and license_kinds")
        license_kinds = [license_kind]

    if _require_license_text_is_a_file:
        srcs = native.glob([license_text])
        if len(srcs) != 1 or srcs[0] != license_text:
            fail("Specified license file doesn't exist: %s" % license_text)

    if namespace:
        print("license(namespace=<str>) is deprecated.")

    _license(
        name = name,
        license_kinds = license_kinds,
        license_text = license_text,
        copyright_notice = copyright_notice,
        package_name = package_name,
        package_url = package_url,
        package_version = package_version,
        applicable_licenses = [],
        visibility = visibility,
        tags = tags,
        testonly = 0,
    )