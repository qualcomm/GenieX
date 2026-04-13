bazel_features = struct(
    cc = struct(
        protobuf_on_allowlist = False,
        swift_fragment_removed = False,
    ),
    external_deps = struct(
        extension_metadata_has_reproducible = True,
        module_extension_has_os_arch_dependent = False,
    ),
    globals = struct(
        PackageSpecificationInfo = None,
        ProtoInfo = None,
        cc_proto_aspect = None,
    ),
    proto = struct(
        starlark_proto_info = True,
    ),
    rules = struct(
        _has_launcher_maker_toolchain = False,
        analysis_tests_can_transition_on_experimental_incompatible_flags = False,
    ),
    toolchains = struct(
        has_use_target_platform_constraints = True,
    ),
)
