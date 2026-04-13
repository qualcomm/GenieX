"""
TTS (Text-to-Speech) wrapper for geniex.
Provides high-level Python API for TTS operations.
"""

import ctypes
import logging
from typing import List, Optional

from .internal.model_loader import ModelLoaderMixin
from .geniex_sdk._lib import _lib
from .geniex_sdk.error import check_error
from .geniex_sdk.tts import (
    ml_TTS,
    ml_TTSConfig,
    ml_TtsCreateInput,
    ml_TtsListAvailableVoicesInput,
    ml_TtsListAvailableVoicesOutput,
    ml_TTSResult,
    ml_TtsSynthesizeInput,
    ml_TtsSynthesizeOutput,
)
from .geniex_sdk.types import ModelConfig, ProfileData

logger = logging.getLogger(__name__)


class TTS(ModelLoaderMixin):
    """Text-to-Speech wrapper."""

    def __init__(
        self,
        model_path: str,
        model_name: Optional[str] = None,
        config: Optional[ModelConfig] = None,
        vocoder_path: Optional[str] = None,
        plugin_id: Optional[str] = None,
        device_id: Optional[str] = None,
    ):
        """
        Create and initialize a TTS instance.

        Args:
            model_path: Path to the model file.
            model_name: Name of the model. If None, uses model_path.
            config: Model configuration. If None, uses default config.
            vocoder_path: Path to the vocoder file. If None, uses default.
            plugin_id: Plugin to use for the model. If None, uses default.
            device_id: Device to use for the model. If None, uses default device.
        """
        if config is None:
            config = ModelConfig()

        if model_name is None:
            model_name = model_path

        if not model_name:
            model_name = model_path

        # C interface initialization
        c_config = config.to_c_struct()

        model_name_bytes = model_name.encode('utf-8')
        model_path_bytes = model_path.encode('utf-8')
        vocoder_path_bytes = vocoder_path.encode('utf-8') if vocoder_path else None
        plugin_id_str = plugin_id if plugin_id else ''
        plugin_id_buf = ctypes.create_string_buffer(plugin_id_str.encode('utf-8'))
        device_id_buf = ctypes.create_string_buffer(device_id.encode('utf-8')) if device_id else None

        model_name_buf = ctypes.create_string_buffer(model_name_bytes)
        model_path_buf = ctypes.create_string_buffer(model_path_bytes)
        vocoder_path_buf = ctypes.create_string_buffer(vocoder_path_bytes) if vocoder_path_bytes else None

        c_input = ml_TtsCreateInput(
            model_name=ctypes.cast(model_name_buf, ctypes.c_char_p),
            model_path=ctypes.cast(model_path_buf, ctypes.c_char_p),
            config=c_config,
            vocoder_path=ctypes.cast(vocoder_path_buf, ctypes.c_char_p) if vocoder_path_buf else None,
            plugin_id=ctypes.cast(plugin_id_buf, ctypes.c_char_p) if plugin_id_buf else None,
            device_id=ctypes.cast(device_id_buf, ctypes.c_char_p) if device_id_buf else None,
        )

        self._string_refs = [
            model_name_buf,
            model_path_buf,
            vocoder_path_buf,
            plugin_id_buf,
            device_id_buf,
        ]

        handle_ptr = ctypes.POINTER(ml_TTS)()
        error_code = _lib.ml_tts_create(ctypes.pointer(c_input), ctypes.byref(handle_ptr))
        check_error(error_code)

        if not handle_ptr:
            raise RuntimeError('Failed to create TTS instance')

        self._handle = handle_ptr
        self._config = config

    def __del__(self):
        """Destroy TTS instance and free associated resources."""
        if hasattr(self, '_handle') and self._handle:
            try:
                _lib.ml_tts_destroy(self._handle)
            except Exception as e:
                logger.warning(f'Error during TTS cleanup: {e}')
            self._handle = None

    @classmethod
    def _get_model_type(cls) -> str:
        return 'tts'

    @classmethod
    def _extract_model_params_from_manifest(
        cls,
        manifest,
        model_path: str,
        repo_id: str,
        store,
        **kwargs,
    ) -> dict:
        """Extract TTS-specific parameters from manifest."""
        model_name = manifest.model_name or repo_id
        if not model_name:
            model_name = repo_id

        vocoder_path = None
        if hasattr(manifest, 'vocoder_file') and manifest.vocoder_file and manifest.vocoder_file.name:
            vocoder_path_obj = store.modelfile_path(repo_id, manifest.vocoder_file.name)
            if vocoder_path_obj.exists():
                vocoder_path = str(vocoder_path_obj)

        plugin_id = manifest.plugin_id.strip() if manifest.plugin_id else None
        if not plugin_id:
            plugin_id = kwargs.pop('plugin_id', None)
        else:
            # Remove plugin_id from kwargs if manifest has it (to avoid passing it twice)
            kwargs.pop('plugin_id', None)
        if plugin_id:
            plugin_id = plugin_id.strip() if isinstance(plugin_id, str) else plugin_id
            if not plugin_id:
                plugin_id = None

        if not plugin_id:
            plugin_id = cls._get_default_plugin_id()

        device_id = manifest.device_id.strip() if manifest.device_id else None
        if not device_id:
            device_id = kwargs.pop('device_id', None)
        if device_id:
            device_id = device_id.strip() if isinstance(device_id, str) else device_id
            if not device_id:
                device_id = None

        return {
            'model_path': str(model_path),
            'model_name': model_name,
            'vocoder_path': vocoder_path,
            'plugin_id': plugin_id,
            'device_id': device_id,
            **kwargs,
        }

    @classmethod
    def _create_instance(cls, **params) -> 'TTS':
        """Create TTS instance with given parameters."""
        return cls(**params)

    @classmethod
    def _get_default_plugin_id(cls) -> str:
        """Get default plugin ID for TTS."""
        return 'cpu_gpu'

    def synthesize(
        self,
        text: str,
        output_path: str,
        voice: Optional[str] = None,
        speed: float = 1.0,
        seed: int = -1,
        sample_rate: int = 22050,
    ) -> 'SynthesizeResult':
        """
        Synthesize text to speech.

        Args:
            text: Text to synthesize.
            output_path: Path to save the audio file.
            voice: Voice identifier. If None, uses default.
            speed: Speech speed multiplier.
            seed: Random seed. -1 for random.
            sample_rate: Audio sample rate.

        Returns:
            SynthesizeResult containing audio file path and metadata.

        Raises:
            Geniex: If synthesis fails.
        """
        # C interface implementation
        tts_config = ml_TTSConfig(
            voice=voice.encode('utf-8') if voice else None,
            speed=ctypes.c_float(speed),
            seed=ctypes.c_int32(seed),
            sample_rate=ctypes.c_int32(sample_rate),
        )

        text_bytes = text.encode('utf-8')
        text_buf = ctypes.create_string_buffer(text_bytes)
        output_path_bytes = output_path.encode('utf-8')
        output_path_buf = ctypes.create_string_buffer(output_path_bytes)

        c_input = ml_TtsSynthesizeInput(
            text_utf8=ctypes.cast(text_buf, ctypes.c_char_p),
            config=ctypes.pointer(tts_config),
            output_path=ctypes.cast(output_path_buf, ctypes.c_char_p),
        )

        c_output = ml_TtsSynthesizeOutput()
        error_code = _lib.ml_tts_synthesize(self._handle, ctypes.pointer(c_input), ctypes.pointer(c_output))
        check_error(error_code)

        return self._extract_result(c_output)

    def _extract_result(self, c_output: ml_TtsSynthesizeOutput) -> 'SynthesizeResult':
        """Extract SynthesizeResult from C output structure."""
        result = c_output.result
        audio_path = ''
        if result.audio_path:
            audio_path = result.audio_path.decode('utf-8')

        profile_data = ProfileData.from_c_struct(c_output.profile_data)
        return SynthesizeResult(
            audio_path=audio_path,
            duration_seconds=result.duration_seconds,
            sample_rate=result.sample_rate,
            channels=result.channels,
            num_samples=result.num_samples,
            profile_data=profile_data,
        )

    def list_available_voices(self) -> List[str]:
        """
        List available voices.

        Returns:
            List of voice identifiers.

        Raises:
            Geniex: If query fails.
        """
        # C interface implementation
        c_input = ml_TtsListAvailableVoicesInput(reserved=None)
        c_output = ml_TtsListAvailableVoicesOutput()

        error_code = _lib.ml_tts_list_available_voices(
            self._handle,
            ctypes.pointer(c_input),
            ctypes.pointer(c_output),
        )
        check_error(error_code)

        voices = []
        if c_output.voice_ids and c_output.voice_count > 0:
            for i in range(c_output.voice_count):
                voice_ptr = c_output.voice_ids[i]
                if voice_ptr:
                    voices.append(voice_ptr.decode('utf-8'))
            _lib.ml_free(c_output.voice_ids)

        return voices


class SynthesizeResult:
    """Result of TTS synthesis."""

    def __init__(
        self,
        audio_path: str,
        duration_seconds: float,
        sample_rate: int,
        channels: int,
        num_samples: int,
        profile_data: ProfileData,
    ):
        self.audio_path = audio_path
        self.duration_seconds = duration_seconds
        self.sample_rate = sample_rate
        self.channels = channels
        self.num_samples = num_samples
        self.profile_data = profile_data

    def __repr__(self) -> str:
        return f'SynthesizeResult(audio_path={self.audio_path}, duration={self.duration_seconds}s, sample_rate={self.sample_rate})'
