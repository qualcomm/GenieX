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

import builtins
import os
import platform
import sys
from datetime import datetime
from io import TextIOWrapper
from pathlib import Path

from . import utils

log_dir = Path('bench-' + datetime.now().strftime('%Y%m%d-%H%M%S'))
log_file: TextIOWrapper


def init():
    global log_file

    os.makedirs(log_dir, exist_ok=True)

    log_file = open(os.path.join(log_dir, 'environment'), 'w', encoding='utf-8')
    print('========== Environment ===========')
    print(f'OS: {platform.system()}')
    print(f'Arch: {platform.machine()}')
    print(f'Python version: {sys.version}')
    res = utils.execute_geniex(['version'])
    if res.returncode != 0:
        raise RuntimeError('Failed to get geniex version')
    for line in res.stdout.strip().splitlines():
        print(line)
    log_file.close()

    log_file = open(log_dir / 'bench.log', 'w', encoding='utf-8')


def print(msg: str):
    global log_file

    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
    data = f'[{ts}] {msg}'
    builtins.print(data)
    log_file.write(data)
    log_file.write('\n')
    log_file.flush()
