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

import subprocess

# Launch PowerShell with the -NoProfile flag which is generally faster to start
# Use -NoExit to keep PowerShell open after running the command
# Use -Command to specify the command to run
subprocess.Popen(
    ["powershell", "-NoProfile", "-NoExit", "-Command", "geniex"],
    creationflags=subprocess.CREATE_NEW_CONSOLE
)
