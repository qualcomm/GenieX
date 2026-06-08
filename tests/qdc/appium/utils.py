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

"""adb + appium helpers for the on-device pytest run (QDC Android phones)."""

from __future__ import annotations

import os
import subprocess
import tempfile

from appium.options.common import AppiumOptions

QDC_LOGS_PATH = '/data/local/tmp/QDC_logs'

options = AppiumOptions()
options.set_capability('automationName', 'UiAutomator2')
options.set_capability('platformName', 'Android')
options.set_capability('deviceName', os.getenv('ANDROID_DEVICE_VERSION'))


def write_qdc_log(filename: str, content: str) -> None:
    """Push content to /data/local/tmp/QDC_logs/<filename> for QDC collection."""
    dest = f'{QDC_LOGS_PATH}/{filename}'
    subprocess.run(['adb', 'shell', f'mkdir -p {os.path.dirname(dest)}'], check=False)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.tmp', delete=False) as f:
        f.write(content)
        tmp = f.name
    try:
        subprocess.run(['adb', 'push', tmp, dest], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    finally:
        os.unlink(tmp)
