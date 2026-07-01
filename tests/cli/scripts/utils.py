# Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

import os
import platform
import shutil
import subprocess
from pathlib import Path
from typing import Any

geniex_path = None


def _search_geniex() -> str:
    search_dirs = [
        "../../../bazel-bin/cli/cmd/geniex/geniex_",
        "./bazel-bin/cli/cmd/geniex/geniex_",
    ]
    for d in search_dirs:
        exe = "geniex" if platform.system() != "Windows" else "geniex.exe"
        path = Path(d) / exe
        if path.exists() and os.access(path, os.X_OK):
            return str(path.resolve())

    global_geniex = shutil.which("geniex")
    if global_geniex is not None:
        return global_geniex

    raise FileNotFoundError("geniex command not found")


def start_geniex(
    args: list[str],
    debug_log: bool = False,
    stdout: Any = subprocess.PIPE,
    stderr: Any = subprocess.PIPE,
    **kwargs: Any,
) -> subprocess.Popen[str]:
    global geniex_path

    if geniex_path is None:
        geniex_path = _search_geniex()

    env = os.environ.copy()
    env["GENIEX_LOG"] = "trace" if debug_log else ""
    env["NO_COLOR"] = "1"

    return subprocess.Popen(
        [geniex_path, "--test-mode", "--skip-update"] + args,
        text=True,
        encoding="utf-8",
        cwd=Path(__file__).parent.parent,
        env=env,
        stdout=stdout,
        stderr=stderr,
        **kwargs,
    )


def execute_geniex(
    args: list[str],
    debug_log: bool = False,
    timeout: int | None = None,
    **kwargs: Any,
) -> subprocess.CompletedProcess[str]:
    proc = start_geniex(args, debug_log=debug_log, **kwargs)
    stdout, stderr = proc.communicate(timeout=timeout)
    return subprocess.CompletedProcess(proc.args, proc.returncode, stdout, stderr)
