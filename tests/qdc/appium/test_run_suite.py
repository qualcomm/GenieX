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

"""Host-side appium entry: build portable Python, deploy, run the suite on-device.

QDC runs this on its appium host (the appium session unlocks adb). The host is a
minimal container with no bash, so deploy is pure-Python adb (see deploy.py): it
builds a portable Python, pushes pkg-geniex + bindings + tests/ to the phone,
and runs the device-gated pytest cells in adb shell. The device-side JUnit is
pulled back to the host; conftest.pytest_sessionfinish ships it to QDC_logs.
"""

from pathlib import Path

import deploy

# This file sits in the artifact's tests/ dir (QDC runs `pytest tests`); the
# deployable mini-repo (incl. the prebuilt termux-usr) is at ../payload.
PAYLOAD = Path(__file__).parents[1] / 'payload'
DEVICE_JUNIT = f'{deploy.DEV_TESTS}/device-results.xml'
HOST_JUNIT = 'device-results.xml'

# api already runs on GitHub; here we want the device-gated cells ("not api").
PYTEST_ARGS = ['-m', 'not api', '--junitxml=device-results.xml']


def test_run_suite():
    deploy.deploy(PAYLOAD)
    rc = deploy.run_pytest(PYTEST_ARGS)
    deploy.adb('pull', DEVICE_JUNIT, HOST_JUNIT, check=False)
    assert Path(HOST_JUNIT).exists(), 'device produced no JUnit XML'
    assert rc == 0, f'on-device pytest exited {rc}'
