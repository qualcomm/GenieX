#!/bin/sh
# Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Pre-flight check for the GenieX CLI on Linux ARM64. Verifies the required
# system libraries and minimum Qualcomm driver versions; intended to be run
# independently of install.sh.
#
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/qcom-ai-hub/geniex/main/cli/release/linux/check.sh | sh

# Wrap the whole script in a brace group so a partial download from
# `curl | sh` cannot execute a truncated tail.
{

set -eu

QUIET=0

# QCOM driver SONAMEs (_QCOM_LIBS in BUILD.bazel) + trixie.yaml runtime libs.
REQUIRED_LIBS="
libCB.so.1
libOpenCL.so.1
libOpenCL_adreno.so.1
libadreno_utils.so.1
libatomic.so.1
libcdsprpc.so.1.0.0
libdmabufheap.so.0.0.0
libglib-2.0.so.0
libgsl.so.1
libllvm-glnext.so.1
libllvm-qcom.so.1
libllvm-qgl.so.1
libpropertyvault.so.0.0.0
"

# Driver probes: `<label>:<lib>:<debug-path-prefix>:<min-ver>`.
DRIVER_PROBES="
GPU (Adreno):libOpenCL.so.1:qcom-adreno:1.838.3
NPU (FastRPC):libcdsprpc.so.1.0.0:fastrpc:15.0
"

usage() {
    cat <<EOF
Pre-flight check for the GenieX CLI on Linux ARM64.

Usage: check.sh [options]

Options:
  -q, --quiet  Suppress non-error output
  -h, --help   Show this help

Exits 0 if all required libraries and driver versions are present.
EOF
}

err() { printf 'error: %s\n' "$*" >&2; }
say() { [ "$QUIET" -eq 1 ] || printf '%s\n' "$*"; }

while [ $# -gt 0 ]; do
    case "$1" in
        -q|--quiet)
            QUIET=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            err "unknown option: $1"
            usage >&2
            exit 2
            ;;
    esac
done

os=$(uname -s 2>/dev/null || echo unknown)
arch=$(uname -m 2>/dev/null || echo unknown)
case "$os" in
    Linux) ;;
    *)
        err "unsupported OS: $os (this checker only supports Linux ARM64)"
        exit 1
        ;;
esac
case "$arch" in
    aarch64|arm64) ;;
    *)
        err "unsupported architecture: $arch (this checker only supports ARM64)"
        exit 1
        ;;
esac

# Locate a shared library by SONAME under /usr/lib. Docker bind-mounts host
# /usr/lib onto /opt/qcom-lib, so /usr/lib is the only directory we expect
# QCOM libs to live in across both bare-metal and container installs.
find_lib() {
    _name="$1"
    if [ -e "/usr/lib/$_name" ]; then
        printf '%s\n' "/usr/lib/$_name"
        return 0
    fi
    return 1
}

# Compare two dotted versions. Returns 0 iff $1 >= $2.
version_ge() {
    [ "$1" = "$2" ] && return 0
    _lo=$(printf '%s\n%s\n' "$1" "$2" | sort -V | head -n1)
    [ "$_lo" = "$2" ]
}

check_required_libs() {
    _missing=""
    for _lib in $REQUIRED_LIBS; do
        find_lib "$_lib" >/dev/null || _missing="${_missing} ${_lib}"
    done
    if [ -n "$_missing" ]; then
        err "missing required shared libraries:"
        for _m in $_missing; do err "  - $_m"; done
        err "install the Qualcomm driver package and Debian: libatomic1 libglib2.0-0t64"
        return 1
    fi
    return 0
}

check_driver_versions() {
    if ! command -v strings >/dev/null 2>&1; then
        say "Note: 'strings' not on PATH — skipping driver version checks"
        return 0
    fi
    _ifs="$IFS"
    IFS='
'
    _failed=0
    for _entry in $DRIVER_PROBES; do
        [ -z "$_entry" ] && continue
        _label=$(printf '%s' "$_entry" | cut -d: -f1)
        _lib=$(printf   '%s' "$_entry" | cut -d: -f2)
        _prefix=$(printf '%s' "$_entry" | cut -d: -f3)
        _min=$(printf    '%s' "$_entry" | cut -d: -f4)
        _path=$(find_lib "$_lib") || continue
        _ver=$(strings -a "$_path" 2>/dev/null \
            | sed -n "s#.*${_prefix}/\([0-9][0-9]*\(\.[0-9][0-9]*\)\{1,2\}\).*#\1#p" \
            | sort -V | tail -n1)
        if [ -z "$_ver" ]; then
            say "Note: could not detect $_label driver version from $_path — continuing"
            continue
        fi
        if version_ge "$_ver" "$_min"; then
            say "$_label driver: $_ver (>= $_min)"
        else
            err "$_label driver $_ver is older than required $_min ($_path)"
            _failed=1
        fi
    done
    IFS="$_ifs"
    if [ "$_failed" -eq 1 ]; then
        err "update the Qualcomm driver packages"
        return 1
    fi
    return 0
}

say "Checking required libraries"
check_required_libs   || exit 1
check_driver_versions || exit 1
say "All checks passed."

exit 0

}  # end of script
