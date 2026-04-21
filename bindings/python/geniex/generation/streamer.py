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

from __future__ import annotations

import queue
import threading
from ctypes import c_bool, c_void_p
from typing import Callable, Iterator

from ..geniex_sdk._types import geniex_token_callback
from .output import GenerateOutput

_SENTINEL = object()


class TextIteratorStreamer:
    """Yields decoded text chunks as the model generates them.

    Usage::

        for chunk in model.generate(prompt, stream=True):
            print(chunk, end='', flush=True)
        # After the loop: streamer.output holds the final GenerateOutput
    """

    def __init__(self) -> None:
        self._queue: queue.Queue = queue.Queue()
        self._output: GenerateOutput | None = None
        self._error: BaseException | None = None
        self._thread: threading.Thread | None = None

    @property
    def output(self) -> GenerateOutput | None:
        return self._output

    def _make_callback(self) -> geniex_token_callback:
        q = self._queue

        @geniex_token_callback
        def _cb(token_bytes: bytes, _userdata: c_void_p) -> c_bool:
            if token_bytes is not None:
                q.put(token_bytes.decode('utf-8', errors='replace'))
            return True

        # Keep a reference so the GC doesn't collect it while C holds the pointer
        self._cb_ref = _cb
        return _cb

    def _run(self, generate_fn: Callable[[], GenerateOutput]) -> None:
        try:
            result = generate_fn()
            self._output = result
        except BaseException as exc:
            self._error = exc
        finally:
            self._queue.put(_SENTINEL)

    def start(self, generate_fn: Callable[[], GenerateOutput]) -> None:
        self._thread = threading.Thread(target=self._run, args=(generate_fn,), daemon=True)
        self._thread.start()

    def __iter__(self) -> Iterator[str]:
        while True:
            item = self._queue.get()
            if item is _SENTINEL:
                break
            yield item
        if self._error is not None:
            raise self._error
        if self._thread is not None:
            self._thread.join()
