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
# Installer for the GenieX CLI on Linux ARM64.
#
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/qcom-ai-hub/geniex/main/cli/release/linux/install.sh | sh
#   curl -fsSL <url> | sh -s -- --version v0.4.0
#   curl -fsSL <url> | sh -s -- --prefix /opt/geniex

# Wrap the whole script in a brace group so a partial download from
# `curl | sh` cannot execute a truncated tail.
{

set -eu

S3_BASE="https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-geniex"
ASSET_STEM="geniex-cli-linux-arm64"
ISSUE_URL="https://github.com/qcom-ai-hub/geniex/issues"

VERSION=""
PREFIX=""
QUIET=0

usage() {
    cat <<EOF
Install the GenieX CLI on Linux ARM64.

Usage: install.sh [options]

Options:
  --version vX.Y.Z   Install a specific release (default: latest stable)
  --prefix DIR       Install to DIR (default: /usr/local/lib/geniex when root,
                     \${XDG_DATA_HOME:-\$HOME/.local/share}/geniex otherwise)
  -q, --quiet        Suppress non-error output
  -h, --help         Show this help

The CLI is symlinked into /usr/local/bin (root) or \$HOME/.local/bin (user).
EOF
}

err() { printf 'error: %s\n' "$*" >&2; }
say() { [ "$QUIET" -eq 1 ] || printf '%s\n' "$*"; }

need_cmd() {
    command -v "$1" >/dev/null 2>&1 || {
        err "missing required command: $1"
        return 1
    }
}

# Pick the first available command from the args; print it on stdout.
pick_cmd() {
    for c in "$@"; do
        if command -v "$c" >/dev/null 2>&1; then
            printf '%s\n' "$c"
            return 0
        fi
    done
    return 1
}

while [ $# -gt 0 ]; do
    case "$1" in
        --version)
            [ $# -ge 2 ] || { err "--version requires an argument"; exit 2; }
            VERSION="$2"
            shift 2
            ;;
        --version=*)
            VERSION="${1#*=}"
            shift
            ;;
        --prefix)
            [ $# -ge 2 ] || { err "--prefix requires an argument"; exit 2; }
            PREFIX="$2"
            shift 2
            ;;
        --prefix=*)
            PREFIX="${1#*=}"
            shift
            ;;
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
        err "unsupported OS: $os (this installer only supports Linux ARM64)"
        err "report at $ISSUE_URL if you need another platform"
        exit 1
        ;;
esac
case "$arch" in
    aarch64|arm64) ;;
    *)
        err "unsupported architecture: $arch (this installer only supports ARM64)"
        err "report at $ISSUE_URL if you need another platform"
        exit 1
        ;;
esac

# Snapdragon EVKs and most Linux container images log in as root, so installing
# under $HOME/.local would leave geniex hidden under /root/.local/bin. Default
# root installs to /usr/local/{lib,bin}/geniex instead, which is on PATH.
IS_ROOT=0
if [ "$(id -u 2>/dev/null || echo 1)" = "0" ]; then
    IS_ROOT=1
fi

need_cmd tar
need_cmd mktemp

DOWNLOADER=$(pick_cmd curl wget) || {
    err "need 'curl' or 'wget' on PATH"
    exit 1
}
HASHER=$(pick_cmd sha256sum shasum) || {
    err "need 'sha256sum' or 'shasum' on PATH"
    exit 1
}

if [ -n "$VERSION" ]; then
    # Allow the user to omit the leading 'v'.
    case "$VERSION" in
        v*) ;;
        *) VERSION="v$VERSION" ;;
    esac
    asset="${ASSET_STEM}-${VERSION}.tar.gz"
else
    asset="${ASSET_STEM}.tar.gz"
fi
url="${S3_BASE}/${asset}"
sha_url="${url}.sha256"

if [ -z "$PREFIX" ]; then
    if [ "$IS_ROOT" -eq 1 ]; then
        PREFIX="/usr/local/lib/geniex"
    else
        PREFIX="${XDG_DATA_HOME:-$HOME/.local/share}/geniex"
    fi
fi
if [ "$IS_ROOT" -eq 1 ]; then
    BIN_DIR="/usr/local/bin"
else
    BIN_DIR="$HOME/.local/bin"
fi

tmp=$(mktemp -d 2>/dev/null) || tmp=$(mktemp -d -t geniex)
trap 'rm -rf "$tmp"' EXIT INT TERM

download() {
    _src="$1"
    _dst="$2"
    case "$DOWNLOADER" in
        curl) curl -fsSL --retry 3 -o "$_dst" "$_src" ;;
        wget) wget -q -O "$_dst" "$_src" ;;
    esac
}

say "Downloading $asset"
if ! download "$url" "$tmp/$asset"; then
    err "download failed: $url"
    if [ -z "$VERSION" ]; then
        err "the latest-stable pointer may not exist yet — try --version vX.Y.Z"
    fi
    exit 1
fi

say "Verifying checksum"
if ! download "$sha_url" "$tmp/${asset}.sha256"; then
    err "checksum download failed: $sha_url"
    exit 1
fi

# The sidecar embeds whatever filename CI hashed (always the versioned one,
# even under the mutable-pointer key), so don't rely on `sha256sum -c` —
# extract the hex digest and compare against a fresh hash of the local file.
expected=$(awk '{print $1; exit}' "$tmp/${asset}.sha256")
case "$HASHER" in
    sha256sum) actual=$(sha256sum    "$tmp/$asset" | awk '{print $1}') ;;
    shasum)    actual=$(shasum -a 256 "$tmp/$asset" | awk '{print $1}') ;;
esac
if [ -z "$expected" ] || [ "$expected" != "$actual" ]; then
    err "checksum mismatch for $asset"
    err "  expected: $expected"
    err "  actual:   $actual"
    exit 1
fi

say "Extracting"
mkdir -p "$tmp/extract"
# CI packs the tarball with a top-level dir matching the versioned stem
# (release.yml `staged="geniex-cli-linux-arm64-<TAG>"`). Strip it so we
# always land on a stable layout regardless of version.
tar -xzf "$tmp/$asset" -C "$tmp/extract" --strip-components=1

if [ ! -f "$tmp/extract/geniex" ]; then
    err "tarball does not contain 'geniex' binary"
    exit 1
fi
chmod +x "$tmp/extract/geniex"

mkdir -p "$(dirname "$PREFIX")"
mkdir -p "$BIN_DIR"

# Atomic-ish swap: move old aside, move new in, drop old. If the new move
# fails we still have .old to restore from.
if [ -e "$PREFIX" ] || [ -L "$PREFIX" ]; then
    rm -rf "${PREFIX}.old"
    mv "$PREFIX" "${PREFIX}.old"
fi
if ! mv "$tmp/extract" "$PREFIX"; then
    err "failed to install to $PREFIX"
    if [ -e "${PREFIX}.old" ]; then
        mv "${PREFIX}.old" "$PREFIX"
    fi
    exit 1
fi
rm -rf "${PREFIX}.old"

# Launcher wrapper instead of a bare symlink: the binary is built without
# an $ORIGIN rpath, so it can't find sibling libgeniex.so / plugins unless
# LD_LIBRARY_PATH is set. The wrapper does that transparently.
cat > "$BIN_DIR/geniex" <<LAUNCHER
#!/bin/sh
GENIEX_PREFIX="$PREFIX"
LD_LIBRARY_PATH="\${GENIEX_PREFIX}\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}" \\
    exec -a geniex "\${GENIEX_PREFIX}/geniex" "\$@"
LAUNCHER
chmod +x "$BIN_DIR/geniex"

say ""
if [ -n "$VERSION" ]; then
    say "Installed: geniex $VERSION"
else
    say "Installed: geniex (latest stable)"
fi
say "  binary:   $PREFIX/geniex"
say "  launcher: $BIN_DIR/geniex"

case ":${PATH-}:" in
    *":$BIN_DIR:"*)
        say ""
        say "Run 'geniex --help' to get started."
        ;;
    *)
        say ""
        say "Add $BIN_DIR to PATH to use 'geniex' directly:"
        say "  echo 'export PATH=\"$BIN_DIR:\$PATH\"' >> ~/.bashrc   # or ~/.zshrc / ~/.profile"
        say "Then 'source' that file or open a new shell."
        ;;
esac

exit 0

}  # end of script
