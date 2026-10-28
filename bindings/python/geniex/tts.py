# SPDX-License-Identifier: BSD-3-Clause
"""Local compiled-bundle TTS. Audio callbacks receive copied float32 PCM bytes."""

import threading
from ctypes import byref, c_void_p, sizeof, string_at
from dataclasses import dataclass

from ._ffi._api import _check, bind_tts, ensure_init
from ._ffi._types import (
    geniex_audio_callback,
    geniex_TtsCreateInput,
    geniex_TtsSynthesizeInput,
    geniex_TtsSynthesizeOutput,
)


@dataclass(frozen=True)
class SpeechResult:
    pcm: bytes
    sample_rate: int
    samples: int
    first_audio_ms: float
    generation_ms: float
    prefill_npu_ms: float
    step_npu_ms: float
    prefill_calls: int
    step_calls: int
    cancelled: bool

    @property
    def rtf(self):
        return self.generation_ms / (1000 * self.samples / self.sample_rate) if self.samples else 0.0


class GenieXTTS:
    """One loaded speech model. Use as a context manager, before geniex.deinit().

    ``device='npu'`` is the only Phonon device. ``verify_npu=True`` additionally
    requires positive prefill and step accelerator times from shared QNN profiling.
    cancel() is thread safe. Do not close from inside an audio callback.
    Select the shared runtime with set_qairt_runtime_path() before SDK init.
    runtime_path is an optional consistency assertion, not an override.
    """

    def __init__(
        self,
        model_path,
        *,
        runtime_path='',
        text_library,
        soc_model,
        language='en',
        device='npu',
        verify_npu=True,
        plugin_id='qairt',
    ):
        ensure_init()
        self._lib = bind_tts()
        self._handle = c_void_p()
        self._lock = threading.Lock()
        self._life = threading.Lock()
        values = [plugin_id, model_path, device, runtime_path, text_library, language, soc_model]
        encoded = [str(value).encode('utf-8') for value in values]
        if any(b'\0' in value for value in encoded):
            raise ValueError('TTS parameters cannot contain NUL')
        args = geniex_TtsCreateInput(sizeof(geniex_TtsCreateInput), int(verify_npu), *encoded)
        _check(self._lib.geniex_tts_create(byref(args), byref(self._handle)))

    def synthesize(self, text, voice='', on_audio=None):
        """Generate speech. on_audio(pcm_bytes, rate) may return False to cancel.

        Callbacks run on this thread. Exceptions cancel generation and are raised
        after native code returns. A cancelled request returns its partial audio.
        """
        if not text.strip() or '\0' in text or '\0' in voice:
            raise ValueError('Expected nonempty text and strings without NUL')
        if not self._lock.acquire(blocking=False):
            raise RuntimeError('TTS handle is busy')
        try:
            if not self._handle:
                raise RuntimeError('TTS handle is closed')
            chunks, errors = [], []

            @geniex_audio_callback
            def callback(pcm, count, rate, _):
                data = string_at(pcm, count * 4)
                chunks.append(data)
                try:
                    return on_audio(data, rate) is not False if on_audio else True
                except BaseException as exc:
                    errors.append(exc)
                    return False

            args = geniex_TtsSynthesizeInput(
                sizeof(geniex_TtsSynthesizeInput), text.encode('utf-8'), voice.encode('utf-8'), callback, None
            )
            output = geniex_TtsSynthesizeOutput(struct_size=sizeof(geniex_TtsSynthesizeOutput))
            status = self._lib.geniex_tts_synthesize(self._handle, byref(args), byref(output))
            if errors:
                raise errors[0]
            if status != -100006:
                _check(status)
            return SpeechResult(
                b''.join(chunks),
                output.sample_rate,
                output.samples,
                output.first_audio_ms,
                output.generation_ms,
                output.prefill_npu_ms,
                output.step_npu_ms,
                output.prefill_calls,
                output.step_calls,
                status == -100006,
            )
        finally:
            self._lock.release()

    def cancel(self):
        with self._life:
            if self._handle:
                _check(self._lib.geniex_tts_cancel(self._handle))

    def close(self):
        with self._lock, self._life:
            if self._handle:
                _check(self._lib.geniex_tts_destroy(self._handle))
                self._handle = c_void_p()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
