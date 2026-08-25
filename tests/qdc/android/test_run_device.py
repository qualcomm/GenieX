# Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Host-side driver: deploy the payload to the phone and run pytest on it."""

from __future__ import annotations

from pathlib import Path

import android_deploy

PAYLOAD = Path(__file__).parents[1] / 'payload'
DEVICE_JUNIT_HOST = 'device-results.xml'
DEVICE_REPORT_HOST = 'device-report.log'
DEVICE_JUNIT_ON_PHONE = f'{android_deploy.DEV_TESTS}/{DEVICE_JUNIT_HOST}'
DEVICE_REPORT_ON_PHONE = f'{android_deploy.DEV_TESTS}/{DEVICE_REPORT_HOST}'

PYTEST_ARGS = [
    '.',
    '-v',
    '-s',
    '--tb=short',
    '-m',
    'llama_cpp or qairt',
    f'--junitxml={DEVICE_JUNIT_HOST}',
    f'--report-log={DEVICE_REPORT_HOST}',
]


def test_run_device():
    android_deploy.deploy(PAYLOAD)
    rc = android_deploy.run_pytest(PYTEST_ARGS)
    # Pull xml/report-log if they exist; host-side run_qdc_pytest.py falls back to
    # parsing the appium stdout when neither survives (Fatal Python + QDC's log
    # collector doesn't reach the device's cwd).
    android_deploy.adb('pull', DEVICE_JUNIT_ON_PHONE, DEVICE_JUNIT_HOST, check=False)
    android_deploy.adb('pull', DEVICE_REPORT_ON_PHONE, DEVICE_REPORT_HOST, check=False)
    assert rc == 0, f'on-device pytest exited {rc}'
