# Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

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
