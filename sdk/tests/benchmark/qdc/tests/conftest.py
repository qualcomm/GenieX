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

"""Appium session fixture for the on-device scorecard run."""

import os

import pytest
from appium import webdriver

from utils import options, write_qdc_log


@pytest.fixture(scope="session", autouse=True)
def driver():
    return webdriver.Remote(
        command_executor="http://127.0.0.1:4723/wd/hub", options=options
    )


def pytest_sessionfinish(session, exitstatus):
    xml = getattr(session.config.option, "xmlpath", None) or "results.xml"
    if os.path.exists(xml):
        with open(xml) as f:
            write_qdc_log("results.xml", f.read())
