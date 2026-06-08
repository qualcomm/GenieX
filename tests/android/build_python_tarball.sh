#!/usr/bin/env bash
# Build a portable Python for Android by extracting Termux's prebuilt aarch64
# packages, then layering pytest + tqdm in via pip on the device. Produces a
# ~22 MB tarball that unpacks under /data/local/tmp and runs directly in adb
# shell — no Termux APK install needed. See tests/android/README.md.
set -euo pipefail

TERMUX_REPO="https://packages.termux.dev/apt/termux-main"
DEPS="python gdbm libandroid-posix-semaphore libandroid-support libbz2 libcrypt libexpat libffi liblzma libsqlite ncurses ncurses-ui-libs openssl readline zlib"
PIP_PKGS="pytest tqdm"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEVICE=""
OUT="$SCRIPT_DIR/dist/python-android-portable.tar.gz"

usage() {
  cat <<EOF
Usage: $(basename "$0") [--device SERIAL] [--out PATH]

Builds python-android-portable.tar.gz from Termux packages. Needs network
access (downloads from packages.termux.dev) and an adb-connected device (pip
runs on-device so wheels match the device's Python).

  --device SERIAL   adb serial (default: first device from 'adb devices')
  --out PATH        output tarball path (default: $OUT)
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --device) DEVICE="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown arg: $1" >&2; usage; exit 1 ;;
  esac
done

if [ -z "$DEVICE" ]; then
  DEVICE="$(adb devices | awk 'NR>1 && $2=="device"{print $1; exit}')"
fi
[ -n "$DEVICE" ] || { echo "no adb device found" >&2; exit 1; }
echo ">> device: $DEVICE"

adb_sh() { adb -s "$DEVICE" shell "$@"; }

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

echo ">> downloading Termux package index"
curl -sL "$TERMUX_REPO/dists/stable/main/binary-aarch64/Packages" -o "$WORK/packages.txt"

echo ">> downloading .deb packages"
mkdir -p "$WORK/deps"
for pkg in $DEPS; do
  fname="$(grep -A 20 "^Package: ${pkg}$" "$WORK/packages.txt" | grep "^Filename:" | head -1 | awk '{print $2}')"
  [ -n "$fname" ] || { echo "package not found in index: $pkg" >&2; exit 1; }
  curl -sL -o "$WORK/deps/$(basename "$fname")" "$TERMUX_REPO/${fname}"
done

echo ">> extracting usr/ from packages"
mkdir -p "$WORK/termux-root"
for deb in "$WORK"/deps/*.deb; do
  ( cd "$WORK/termux-root" && ar x "$deb" && tar -xJf data.tar.xz && rm -f control.tar.xz data.tar.xz debian-binary )
done

echo ">> pushing base usr/ to device"
( cd "$WORK/termux-root/data/data/com.termux/files" && tar -czf "$WORK/termux-usr-base.tar.gz" usr )
adb -s "$DEVICE" push "$WORK/termux-usr-base.tar.gz" /data/local/tmp/ >/dev/null
adb_sh "cd /data/local/tmp && rm -rf termux-usr usr && tar -xzf termux-usr-base.tar.gz && mv usr termux-usr && rm termux-usr-base.tar.gz"

echo ">> installing $PIP_PKGS on-device"
adb_sh "
set -e
export PREFIX=/data/local/tmp/termux-usr
export LD_LIBRARY_PATH=\$PREFIX/lib
export HOME=/data/local/tmp TMPDIR=/data/local/tmp/tmp
mkdir -p \$TMPDIR
\$PREFIX/bin/python3 -m ensurepip
\$PREFIX/bin/python3 -m pip install --target=\$PREFIX/lib/python3.13/site-packages $PIP_PKGS
"

echo ">> packing portable tarball"
adb_sh "cd /data/local/tmp && tar -czf python-android-portable.tar.gz termux-usr"
mkdir -p "$(dirname "$OUT")"
adb -s "$DEVICE" pull /data/local/tmp/python-android-portable.tar.gz "$OUT" >/dev/null
adb_sh "rm -f /data/local/tmp/python-android-portable.tar.gz"

echo ">> done: $OUT ($(du -h "$OUT" | cut -f1))"
