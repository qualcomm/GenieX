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

"""Default progress rendering for model downloads.

Built on :mod:`tqdm.auto`, which picks a notebook widget inside Jupyter
and a plain TTY bar everywhere else. One bar per file, keyed by name.
"""

from __future__ import annotations

from typing import Callable

from tqdm.auto import tqdm

from .model_manager import FileProgress, ProgressCallback

__all__ = ['default_progress_printer', 'resolve', 'finish']


class _TqdmProgress:
    def __init__(self) -> None:
        self._bars: dict[str, tqdm] = {}

    def __call__(self, files: list[FileProgress]) -> bool:
        for f in files:
            bar = self._bars.get(f.file_name)
            if bar is None:
                bar = tqdm(
                    total=f.total_bytes if f.total_bytes > 0 else None,
                    unit='B',
                    unit_scale=True,
                    unit_divisor=1024,
                    desc=f.file_name,
                    leave=True,
                )
                self._bars[f.file_name] = bar
            elif f.total_bytes > 0 and bar.total != f.total_bytes:
                bar.total = f.total_bytes
                bar.refresh()
            bar.update(f.downloaded_bytes - bar.n)
        return True

    def finish(self) -> None:
        for bar in self._bars.values():
            bar.close()
        self._bars.clear()


def default_progress_printer() -> ProgressCallback:
    """Return a ``tqdm.auto``-backed progress callback."""
    return _TqdmProgress()


def resolve(progress: ProgressCallback | bool | None) -> ProgressCallback | None:
    """Normalize the ``progress`` argument shared by public factories.

    ``None`` → default, ``False`` → muted, callable → passthrough.
    """
    if progress is False:
        return None
    if progress is None:
        return default_progress_printer()
    if not callable(progress):
        raise TypeError(f'progress must be callable, False, or None; got {type(progress).__name__}')
    return progress


def finish(printer: Callable[..., bool] | None) -> None:
    """Call ``printer.finish()`` if present; safe on arbitrary callables."""
    if printer is None:
        return
    fn = getattr(printer, 'finish', None)
    if callable(fn):
        fn()
