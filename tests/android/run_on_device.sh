#!/usr/bin/env bash
# Deploy the geniex Python bindings + SDK + tests/ to an Android device and run
# the pytest suite there via adb shell. Reproduces the #682 harness. SDK build,
# model pull, and proxy are out of scope — see tests/android/README.md.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

PKG="$REPO_ROOT/sdk/pkg-geniex"
PY_TARBALL="$SCRIPT_DIR/dist/python-android-portable.tar.gz"

DEV_ROOT="/data/local/tmp"
DEV_PY="$DEV_ROOT/termux-usr"
DEV_SDK="$DEV_ROOT/geniex"
DEV_TESTS="$DEV_ROOT/sdk-tests"
PREFIX="$DEV_PY"

DEVICE=""
FORCE_PYTHON=0

usage() {
  cat <<EOF
Usage: $(basename "$0") [--device SERIAL] <command> [args]

Commands:
  deploy [--force-python]   Push portable Python, pkg-geniex, bindings, tests/
                            to the device and flatten plugin libs (idempotent).
  test -- <pytest args>     Run pytest on-device with the SDK env. No default
                            selection — you must pass -m / paths, e.g.
                              test -- -m api --ignore=api/test_device_list.py
                              test -- plugins/qairt/test_qairt_llm.py
  shell                     Open an interactive adb shell with the SDK env set.

Options:
  --device SERIAL           adb serial (default: first device from 'adb devices')
EOF
}

[ $# -gt 0 ] || { usage; exit 1; }
while [ $# -gt 0 ]; do
  case "$1" in
    --device) DEVICE="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    deploy|test|shell) CMD="$1"; shift; break ;;
    *) echo "unknown arg: $1" >&2; usage; exit 1 ;;
  esac
done
: "${CMD:?no command given}"

if [ -z "$DEVICE" ]; then
  DEVICE="$(adb devices | awk 'NR>1 && $2=="device"{print $1; exit}')"
fi
[ -n "$DEVICE" ] || { echo "no adb device found" >&2; exit 1; }

adb_sh() { adb -s "$DEVICE" shell "$@"; }
adb_push() { adb -s "$DEVICE" push "$@" >/dev/null; }

# Same env the #682 reproduction used. qairt loads GENIEX_LIB_PATH/libQnn*.so
# and the llama_cpp Hexagon FastRPC layer resolves ggml-htp skels off
# LD_LIBRARY_PATH — both expect a flat lib/, which `deploy` arranges.
device_env() {
  cat <<EOF
export PREFIX=$PREFIX
export LD_LIBRARY_PATH=\$PREFIX/lib:$DEV_SDK/lib:$DEV_SDK/lib/qairt:$DEV_SDK/lib/qairt/htp-files:/system/lib64
export GENIEX_LIB_PATH=$DEV_SDK/lib
export HOME=$DEV_ROOT
export GENIEX_DEVICE_TEST=1
EOF
}

cmd_deploy() {
  while [ $# -gt 0 ]; do
    case "$1" in
      --force-python) FORCE_PYTHON=1; shift ;;
      *) echo "unknown deploy arg: $1" >&2; exit 1 ;;
    esac
  done

  [ -f "$PKG/lib/libgeniex.so" ] || {
    echo "missing $PKG/lib/libgeniex.so — build the Android SDK first (/build)" >&2
    exit 1
  }

  # 1. portable Python (one-time per device unless --force-python)
  if [ "$FORCE_PYTHON" = 1 ] || ! adb_sh "test -x $DEV_PY/bin/python3" 2>/dev/null; then
    [ -f "$PY_TARBALL" ] || {
      echo "missing $PY_TARBALL — run build_python_tarball.sh first" >&2
      exit 1
    }
    echo ">> deploying portable Python"
    adb_push "$PY_TARBALL" "$DEV_ROOT/"
    adb_sh "cd $DEV_ROOT && rm -rf termux-usr && tar -xzf python-android-portable.tar.gz && rm python-android-portable.tar.gz"
  else
    echo ">> portable Python already present (use --force-python to redeploy)"
  fi

  # 2. pkg-geniex
  echo ">> deploying pkg-geniex"
  local tmp; tmp="$(mktemp -d)"
  tar -czf "$tmp/pkg-geniex.tar.gz" -C "$REPO_ROOT/sdk" pkg-geniex
  adb_push "$tmp/pkg-geniex.tar.gz" "$DEV_ROOT/"
  adb_sh "cd $DEV_ROOT && rm -rf geniex && tar -xzf pkg-geniex.tar.gz && mv pkg-geniex geniex && rm pkg-geniex.tar.gz"

  # 3. flatten plugin libs into lib/ (SDK bugs #3 + Hexagon ADSP path, see #682)
  echo ">> flattening plugin libs into lib/"
  adb_sh "
cd $DEV_SDK/lib
for f in qairt/htp-files/*.so qairt/htp-files/*.cat llama_cpp/*.so; do
  bn=\$(basename \"\$f\")
  [ \"\$bn\" != libgeniex.so ] && [ -e \"\$f\" ] && ln -sf \"\$f\" \"\$bn\"
done
true
"

  # 4. tests/ (without models/ and android/)
  echo ">> deploying tests/"
  tar --exclude='tests/models' --exclude='tests/android' -czf "$tmp/sdk-tests.tar.gz" -C "$REPO_ROOT" tests
  adb_push "$tmp/sdk-tests.tar.gz" "$DEV_ROOT/"
  adb_sh "cd $DEV_ROOT && rm -rf sdk-tests && tar -xzf sdk-tests.tar.gz && mv tests sdk-tests && rm sdk-tests.tar.gz"

  # 5. geniex Python bindings
  echo ">> deploying geniex Python bindings"
  tar -czf "$tmp/geniex-py.tar.gz" -C "$REPO_ROOT/bindings/python" geniex _geniex_backend.py _sdk_fetch.py
  adb_push "$tmp/geniex-py.tar.gz" "$DEV_ROOT/"
  adb_sh "cd $DEV_PY/lib/python3.13/site-packages && rm -rf geniex && tar -xzf $DEV_ROOT/geniex-py.tar.gz && rm $DEV_ROOT/geniex-py.tar.gz"

  rm -rf "$tmp"
  echo ">> deploy complete"
}

cmd_test() {
  [ "${1:-}" = "--" ] && shift
  [ $# -gt 0 ] || { echo "no pytest args — pass -m / paths after '--'" >&2; exit 1; }
  local args=""
  for a in "$@"; do
    args="$args '${a//\'/\'\\\'\'}'"
  done
  adb_sh "
$(device_env)
cd $DEV_TESTS
\$PREFIX/bin/python3 -m pytest$args
"
}

cmd_shell() {
  adb -s "$DEVICE" shell "
$(device_env)
cd $DEV_TESTS
exec sh
"
}

case "$CMD" in
  deploy) cmd_deploy "$@" ;;
  test) cmd_test "$@" ;;
  shell) cmd_shell ;;
esac
