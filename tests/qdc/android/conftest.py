# Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Host-side pytest fixtures for the QDC Android leg.

Creating the APPIUM driver in a session-scoped autouse fixture is what
unlocks adb to the phone; without it, `adb` on the QDC appium container
can't see the device.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

import pytest
from appium import webdriver
from appium.options.common import AppiumOptions

DEVICE_JUNIT = 'device-results.xml'
DEVICE_REPORT = 'device-report.log'
QDC_LOGS = '/data/local/tmp/QDC_logs'


def _adb(*args: str) -> None:
    subprocess.run(['adb', *args], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)


def _prepare_device_for_appium() -> None:
    # SM8850's verify-apps blocks io.appium.settings on install with
    # INSTALL_FAILED_VERIFICATION_FAILURE. Disabling verifier_verify_adb_installs
    # + removing any stale package makes the UiAutomator2 driver's session-start
    # install succeed. `settings put` and `pm uninstall` are no-ops when they
    # already reflect the desired state, so the fixture is idempotent.
    _adb('shell', 'settings', 'put', 'global', 'verifier_verify_adb_installs', '0')
    _adb('shell', 'settings', 'put', 'global', 'package_verifier_enable', '0')
    _adb('shell', 'pm', 'uninstall', 'io.appium.settings')
    _adb('shell', 'pm', 'uninstall', 'io.appium.uiautomator2.server')
    _adb('shell', 'pm', 'uninstall', 'io.appium.uiautomator2.server.test')


def _options() -> AppiumOptions:
    opts = AppiumOptions()
    opts.set_capability('automationName', 'UiAutomator2')
    opts.set_capability('platformName', 'Android')
    opts.set_capability('deviceName', os.getenv('ANDROID_DEVICE_VERSION'))
    # UiAutomator2 pushes its io.appium.settings helper APK on session start;
    # 60s is tight when QDC's shared adb daemon is loaded, so bump to 300s.
    opts.set_capability('appium:adbExecTimeout', 300000)
    return opts


@pytest.fixture(scope='session', autouse=True)
def driver():
    _prepare_device_for_appium()
    return webdriver.Remote(command_executor='http://127.0.0.1:4723/wd/hub', options=_options())


def pytest_sessionfinish(session, exitstatus):
    to_push = [name for name in (DEVICE_JUNIT, DEVICE_REPORT) if Path(name).exists()]
    if not to_push:
        return
    subprocess.run(['adb', 'shell', f'mkdir -p {QDC_LOGS}'], check=False)
    for name in to_push:
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(Path(name).read_bytes())
            tmp = f.name
        try:
            subprocess.run(['adb', 'push', tmp, f'{QDC_LOGS}/{name}'])
        finally:
            os.unlink(tmp)
