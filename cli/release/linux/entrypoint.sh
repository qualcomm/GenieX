#!/bin/sh
# Resolve host QCOM libs at container start.
#
# /opt/qcom-lib is the user's bind-mounted host /usr/lib. Host layout
# varies: EVK images put libs at the top level; stock Ubuntu puts them
# in aarch64-linux-gnu/. Each known QCOM lib is symlinked into
# /opt/geniex (already on LD_LIBRARY_PATH).
#
# install.sh::resolve_host_lib does the equivalent on bare metal.
set -e

LIB_DIR=/opt/geniex
MISSING_LIBS=""

link_lib() {
    bare="$1"
    dest="$LIB_DIR/$bare"
    [ -e "$dest" ] && return 0
    for d in /opt/qcom-lib/aarch64-linux-gnu /opt/qcom-lib; do
        f=$(find "$d" -maxdepth 1 -name "${bare}*" 2>/dev/null \
              | sort -V | tail -n1)
        if [ -n "$f" ] && [ -e "$f" ]; then
            ln -sf "$f" "$dest"
            return 0
        fi
    done
    MISSING_LIBS="$MISSING_LIBS $bare"
}

# fastrpc: bare-name links required by libQnnHtp*Stub.so's $ORIGIN rpath.
link_lib libcdsprpc.so
link_lib libadsprpc.so

# Other QCOM user-space libs (GPU/CL/llvm).
for lib in \
    libOpenCL.so.1 \
    libOpenCL_adreno.so.1 \
    libCB.so.1 \
    libadreno_utils.so.1 \
    libgsl.so.1 \
    libllvm-qcom.so.1 \
    libllvm-qgl.so.1 \
    libllvm-glnext.so.1 \
    libpropertyvault.so.0 \
    libdmabufheap.so.0 \
; do
    link_lib "$lib"
done

if [ -n "$MISSING_LIBS" ]; then
    echo "warning: the following libraries were not found under /opt/qcom-lib; NPU/GPU may fail:" >&2
    for lib in $MISSING_LIBS; do
        echo "  - $lib" >&2
    done
    echo "  install the Qualcomm driver packages on the host (qcom-adreno1, qcom-fastrpc1)." >&2
    echo "  See the GenieX FAQ \"Linux ARM64 setup\" section." >&2
fi

# Default to an interactive bash shell when invoked with no arguments.
if [ "$#" -eq 0 ]; then
    exec /bin/bash
fi
exec "$@"
