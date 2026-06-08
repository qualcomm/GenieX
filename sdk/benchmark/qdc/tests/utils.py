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

"""adb + appium helpers for the on-device scorecard run (QDC Android phones)."""

from __future__ import annotations

import os
import subprocess
import tempfile

from appium.options.common import AppiumOptions

# QDC extracts the artifact under /qdc/appium/ on the runner host.
HOST_BUNDLE = "/qdc/appium/pkg-geniex"
HOST_ROWS = "/qdc/appium/matrix_rows.txt"
HOST_IMAGE = "/qdc/appium/test.png"
BUNDLE_PATH = "/data/local/tmp/pkg-geniex"
MODELS_PATH = "/data/local/tmp/models"
IMAGE_PATH = "/data/local/tmp/test.png"
QDC_LOGS_PATH = "/data/local/tmp/QDC_logs"
RESULTS_PATH = f"{QDC_LOGS_PATH}/results"

options = AppiumOptions()
options.set_capability("automationName", "UiAutomator2")
options.set_capability("platformName", "Android")
options.set_capability("deviceName", os.getenv("ANDROID_DEVICE_VERSION"))


def run_adb_command(cmd: str, *, check: bool = True) -> subprocess.CompletedProcess:
    """Run a command on-device via ``adb shell`` with an exit-code sentinel."""
    raw = subprocess.run(
        ["adb", "shell", f"{cmd}; echo __RC__:$?"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        errors="replace",
    )
    stdout, rc = raw.stdout, raw.returncode
    lines = stdout.rstrip("\n").split("\n") if stdout else []
    if lines and lines[-1].startswith("__RC__:"):
        try:
            rc = int(lines[-1][7:])
            stdout = "\n".join(lines[:-1]) + "\n"
        except ValueError:
            pass
    print(stdout)
    result = subprocess.CompletedProcess(raw.args, rc, stdout=stdout)
    if check:
        assert rc == 0, f"Command failed (exit {rc}): {cmd}"
    return result


def push_bundle_if_needed() -> None:
    """Push pkg-geniex to the device once, making bin/ executable."""
    check = subprocess.run(
        ["adb", "shell", f"ls {BUNDLE_PATH}/bin/geniex_benchmark"],
        text=True,
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if check.returncode != 0:
        subprocess.run(["adb", "push", HOST_BUNDLE, BUNDLE_PATH], check=True)
        run_adb_command(f"find {BUNDLE_PATH}/bin -type f -exec chmod 755 {{}} +")
        # Android expects a flat layout: the qairt plugin loads
        # GENIEX_PLUGIN_PATH/libQnnHtp.so directly, and the Hexagon FastRPC layer
        # resolves the llama_cpp ggml-htp skels via ADSP_LIBRARY_PATH=lib. The CLI
        # package nests both under subdirs, so copy them up to lib/.
        run_adb_command(f"cp {BUNDLE_PATH}/lib/qairt/htp-files/*.so {BUNDLE_PATH}/lib/")
        run_adb_command(f"cp {BUNDLE_PATH}/lib/llama_cpp/*.so {BUNDLE_PATH}/lib/")


def write_qdc_log(filename: str, content: str) -> None:
    """Push content to /data/local/tmp/QDC_logs/<filename> for QDC collection."""
    run_adb_command(f"mkdir -p {os.path.dirname(f'{QDC_LOGS_PATH}/{filename}')}")
    with tempfile.NamedTemporaryFile(mode="w", suffix=".tmp", delete=False) as f:
        f.write(content)
        tmp = f.name
    try:
        subprocess.run(
            ["adb", "push", tmp, f"{QDC_LOGS_PATH}/{filename}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
    finally:
        os.unlink(tmp)
