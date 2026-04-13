"""
Diarize wrapper for geniex.
Provides high-level Python API for speaker diarization operations.
"""

import ctypes
import logging
from typing import List, Optional

from .internal.model_loader import ModelLoaderMixin
from .geniex_sdk._lib import _lib
from .geniex_sdk.diarize import (
    ml_Diarize,
    ml_DiarizeConfig,
    ml_DiarizeCreateInput,
    ml_DiarizeInferInput,
    ml_DiarizeInferOutput,
)
from .geniex_sdk.error import check_error
from .geniex_sdk.types import ModelConfig, ProfileData

logger = logging.getLogger(__name__)


class Diarize(ModelLoaderMixin):
    """Speaker diarization wrapper."""

    def __init__(
        self,
        model_path: str,
        model_name: Optional[str] = None,
        config: Optional[ModelConfig] = None,
        plugin_id: Optional[str] = None,
        device_id: Optional[str] = None,
        license_id: Optional[str] = None,
        license_key: Optional[str] = None,
    ):
        """
        Create and initialize a Diarize instance.

        Args:
            model_path: Path to the model file.
            model_name: Name of the model. If None, uses model_path.
            config: Model configuration. If None, uses default config.
            plugin_id: Plugin to use for the model. If None, uses default.
            device_id: Device to use for the model. If None, uses default device.
            license_id: License ID for loading NPU models.
            license_key: License key for loading NPU models.
        """
        if config is None:
            config = ModelConfig()

        if model_name is None:
            model_name = model_path

        if not model_name:
            model_name = model_path

        c_config = config.to_c_struct()

        model_name_bytes = model_name.encode('utf-8')
        model_path_bytes = model_path.encode('utf-8')
        plugin_id_str = plugin_id if plugin_id else ''
        plugin_id_buf = ctypes.create_string_buffer(plugin_id_str.encode('utf-8'))
        device_id_buf = ctypes.create_string_buffer(device_id.encode('utf-8')) if device_id else None
        license_id_buf = ctypes.create_string_buffer(license_id.encode('utf-8')) if license_id else None
        license_key_buf = ctypes.create_string_buffer(license_key.encode('utf-8')) if license_key else None

        model_name_buf = ctypes.create_string_buffer(model_name_bytes)
        model_path_buf = ctypes.create_string_buffer(model_path_bytes)

        c_input = ml_DiarizeCreateInput(
            model_name=ctypes.cast(model_name_buf, ctypes.c_char_p),
            model_path=ctypes.cast(model_path_buf, ctypes.c_char_p),
            config=c_config,
            plugin_id=ctypes.cast(plugin_id_buf, ctypes.c_char_p) if plugin_id_buf else None,
            device_id=ctypes.cast(device_id_buf, ctypes.c_char_p) if device_id_buf else None,
            license_id=ctypes.cast(license_id_buf, ctypes.c_char_p) if license_id_buf else None,
            license_key=ctypes.cast(license_key_buf, ctypes.c_char_p) if license_key_buf else None,
        )

        self._string_refs = [
            model_name_buf,
            model_path_buf,
            plugin_id_buf,
            device_id_buf,
            license_id_buf,
            license_key_buf,
        ]

        handle_ptr = ctypes.POINTER(ml_Diarize)()
        error_code = _lib.ml_diarize_create(ctypes.pointer(c_input), ctypes.byref(handle_ptr))
        check_error(error_code)

        if not handle_ptr:
            raise RuntimeError('Failed to create Diarize instance')

        self._handle = handle_ptr
        self._config = config

    def __del__(self):
        """Destroy Diarize instance and free associated resources."""
        if hasattr(self, '_handle') and self._handle:
            try:
                _lib.ml_diarize_destroy(self._handle)
            except Exception as e:
                logger.warning(f'Error during Diarize cleanup: {e}')
            self._handle = None

    @classmethod
    def _get_model_type(cls) -> str:
        return 'diarize'

    @classmethod
    def _extract_model_params_from_manifest(
        cls,
        manifest,
        model_path: str,
        repo_id: str,
        store,
        **kwargs,
    ) -> dict:
        """Extract Diarize-specific parameters from manifest."""
        model_name = manifest.model_name or repo_id
        if not model_name:
            model_name = repo_id

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
            'plugin_id': plugin_id,
            'device_id': device_id,
            **kwargs,
        }

    @classmethod
    def _create_instance(cls, **params) -> 'Diarize':
        """Create Diarize instance with given parameters."""
        return cls(**params)

    @classmethod
    def _get_default_plugin_id(cls) -> str:
        """Get default plugin ID for Diarize."""
        return 'qnn'

    def infer(
        self,
        audio_path: str,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
    ) -> 'DiarizeResult':
        """
        Perform speaker diarization on audio file.

        Args:
            audio_path: Path to the audio file.
            min_speakers: Minimum number of speakers. If None, uses default.
            max_speakers: Maximum number of speakers. If None, uses default.

        Returns:
            DiarizeResult containing speech segments and metadata.

        Raises:
            Geniex: If diarization fails.
        """
        diarize_config = ml_DiarizeConfig(
            min_speakers=ctypes.c_int32(min_speakers) if min_speakers is not None else ctypes.c_int32(0),
            max_speakers=ctypes.c_int32(max_speakers) if max_speakers is not None else ctypes.c_int32(0),
        )

        audio_path_bytes = audio_path.encode('utf-8')
        audio_path_buf = ctypes.create_string_buffer(audio_path_bytes)

        c_input = ml_DiarizeInferInput(
            audio_path=ctypes.cast(audio_path_buf, ctypes.c_char_p),
            config=ctypes.pointer(diarize_config),
        )

        c_output = ml_DiarizeInferOutput()
        error_code = _lib.ml_diarize_infer(self._handle, ctypes.pointer(c_input), ctypes.pointer(c_output))
        check_error(error_code)

        return self._extract_result(c_output)

    def _extract_result(self, c_output: ml_DiarizeInferOutput) -> 'DiarizeResult':
        """Extract DiarizeResult from C output structure."""
        segments = []
        if c_output.segments and c_output.segment_count > 0:
            for i in range(c_output.segment_count):
                seg = c_output.segments[i]
                speaker_label = ''
                if seg.speaker_label:
                    speaker_label = seg.speaker_label.decode('utf-8')
                segments.append(
                    SpeechSegment(
                        start_time=seg.start_time,
                        end_time=seg.end_time,
                        speaker_label=speaker_label,
                    )
                )

        profile_data = ProfileData.from_c_struct(c_output.profile_data)
        return DiarizeResult(
            segments=segments,
            num_speakers=c_output.num_speakers,
            duration=c_output.duration,
            profile_data=profile_data,
        )


class SpeechSegment:
    """Speech segment with speaker information."""

    def __init__(self, start_time: float, end_time: float, speaker_label: str):
        self.start_time = start_time
        self.end_time = end_time
        self.speaker_label = speaker_label

    def __repr__(self) -> str:
        return f'SpeechSegment(start={self.start_time:.2f}s, end={self.end_time:.2f}s, speaker={self.speaker_label})'


class DiarizeResult:
    """Result of speaker diarization."""

    def __init__(
        self,
        segments: List[SpeechSegment],
        num_speakers: int,
        duration: float,
        profile_data: ProfileData,
    ):
        self.segments = segments
        self.num_speakers = num_speakers
        self.duration = duration
        self.profile_data = profile_data

    def __repr__(self) -> str:
        return (
            f'DiarizeResult(segments={len(self.segments)}, speakers={self.num_speakers}, duration={self.duration:.2f}s)'
        )
