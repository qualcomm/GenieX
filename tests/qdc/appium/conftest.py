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

"""Appium session fixture for the on-device pytest run.

QDC's APPIUM framework runs this harness on its host; the appium session is what
unlocks ``adb`` access to the attached phone. ``test_run_suite`` then drives the
device and pulls the device-side JUnit to the host as ``device-results.xml``. On
finish we push that into QDC_logs/results so run_qdc_pytest.py can collect it —
the device suite's result is the signal, not this outer harness's own XML.
"""

import os

import pytest
from appium import webdriver
from utils import options, write_qdc_log

DEVICE_JUNIT = 'device-results.xml'


@pytest.fixture(scope='session', autouse=True)
def driver():
    return webdriver.Remote(command_executor='http://127.0.0.1:4723/wd/hub', options=options)


def pytest_sessionfinish(session, exitstatus):
    if os.path.exists(DEVICE_JUNIT):
        with open(DEVICE_JUNIT) as f:
            write_qdc_log('results/device-results.xml', f.read())
    # Always ship the harness log so a failed build/deploy/test is diagnosable.
    if os.path.exists('harness.log'):
        with open('harness.log') as f:
            write_qdc_log('harness.log', f.read())
