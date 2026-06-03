"""go_cgo_test: go_test that can find geniex.dll under `bazel test` on Windows.

The cgo test binary imports geniex.dll via the PE import table, so the loader
needs to find it before any Go code runs. The bazel test runner doesn't put
runfiles on the DLL search path, so on Windows we wrap the test in a .bat
shim that prepends $TEST_SRCDIR/_main/sdk/pkg-geniex/lib to PATH before
exec'ing the real test executable. On Linux the SDK .so is already located
via -Wl,-rpath,. (see //sdk:local_sdk_lib), so this macro degrades to a plain
go_test there.
"""

load("@io_bazel_rules_go//go:def.bzl", "go_test")

_BAT = """@echo off
set "PATH=%TEST_SRCDIR%\\_main\\sdk\\pkg-geniex\\lib;%PATH%"
"%TEST_SRCDIR%\\_main\\{inner}" %*
set "INNER_RC=%ERRORLEVEL%"
rem rules_go writes per-test LCOV fragments to %COVERAGE_DIR%\\go_coverage.*.dat,
rem but bazel only collects the outer test's %COVERAGE_OUTPUT_FILE%. Concat the
rem fragments so the outer LCOV is non-empty.
if defined COVERAGE_OUTPUT_FILE if defined COVERAGE_DIR (
  if exist "%COVERAGE_DIR%\\go_coverage.*.dat" (
    copy /b /y "%COVERAGE_DIR%\\go_coverage.*.dat" "%COVERAGE_OUTPUT_FILE%" >nul
  )
)
exit /b %INNER_RC%
"""

def _bat_wrapper_impl(ctx):
    bat = ctx.actions.declare_file(ctx.label.name + ".bat")
    inner_path = ctx.executable.inner.short_path.replace("/", "\\")
    ctx.actions.write(
        output = bat,
        content = _BAT.format(inner = inner_path),
        is_executable = True,
    )
    runfiles = ctx.runfiles(files = [ctx.executable.inner]).merge(
        ctx.attr.inner[DefaultInfo].default_runfiles,
    )
    return [DefaultInfo(executable = bat, runfiles = runfiles)]

_bat_wrapper_test = rule(
    implementation = _bat_wrapper_impl,
    test = True,
    attrs = {
        "inner": attr.label(mandatory = True, executable = True, cfg = "target"),
    },
)

def go_cgo_test(name, **kwargs):
    """A go_test that injects sdk/pkg-geniex/lib into PATH on Windows.

    Generates two test targets and a test_suite:
      - <name>_posix:   plain go_test, only compatible on non-Windows
      - <name>_windows: bat-wrapper around the inner go_test, only on Windows
      - <name>:         test_suite that picks the right one per platform
    """
    tags = list(kwargs.pop("tags", []))
    posix = name + "_posix"
    windows_inner = name + "_windows_inner"
    windows = name + "_windows"

    # Linux/macOS: plain go_test, marked incompatible on Windows.
    go_test(
        name = posix,
        tags = tags,
        target_compatible_with = select({
            "@platforms//os:windows": ["@platforms//:incompatible"],
            "//conditions:default": [],
        }),
        **kwargs
    )

    # Windows: build the test, then wrap it with a .bat that fixes PATH.
    go_test(
        name = windows_inner,
        tags = tags + ["manual"],
        target_compatible_with = ["@platforms//os:windows"],
        **kwargs
    )

    _bat_wrapper_test(
        name = windows,
        inner = ":" + windows_inner,
        tags = tags,
        target_compatible_with = ["@platforms//os:windows"],
    )

    native.test_suite(
        name = name,
        tests = [":" + posix, ":" + windows],
        tags = tags,
    )
