LicenseKindInfo = provider(
    doc = "Minimal provider for license kind metadata.",
    fields = {
        "conditions": "List of usage conditions.",
        "label": "Label of the license kind target.",
        "long_name": "Human readable license kind name.",
        "name": "Canonical license kind name.",
    },
)

LicenseInfo = provider(
    doc = "Minimal provider for package license metadata.",
    fields = {
        "copyright_notice": "Copyright notice.",
        "label": "Label of the license target.",
        "license_kinds": "List of LicenseKindInfo providers.",
        "license_text": "License file.",
        "package_name": "Package name.",
        "package_url": "Package URL.",
        "package_version": "Package version.",
    },
)