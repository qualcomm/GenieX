"""
Type definitions for nexa-sdk.
Provides Python classes corresponding to C structures.
"""

import ctypes
from dataclasses import dataclass, field
from typing import List, Optional

from .llm import (
    ml_GenerationConfig,
    ml_KvCacheLoadInput,
    ml_KvCacheSaveInput,
    ml_LlmChatMessage,
    ml_ModelConfig,
    ml_ProfileData,
    ml_SamplerConfig,
)


@dataclass
class SamplerConfig:
    """Text generation sampling parameters."""

    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 40
    min_p: float = 0.0
    repetition_penalty: float = 1.0
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    seed: int = -1
    grammar_path: Optional[str] = None
    grammar_string: Optional[str] = None
    enable_json: bool = False

    def to_c_struct(self) -> ml_SamplerConfig:
        """Convert to C structure."""
        return ml_SamplerConfig(
            temperature=ctypes.c_float(self.temperature),
            top_p=ctypes.c_float(self.top_p),
            top_k=ctypes.c_int32(self.top_k),
            min_p=ctypes.c_float(self.min_p),
            repetition_penalty=ctypes.c_float(self.repetition_penalty),
            presence_penalty=ctypes.c_float(self.presence_penalty),
            frequency_penalty=ctypes.c_float(self.frequency_penalty),
            seed=ctypes.c_int32(self.seed),
            grammar_path=self.grammar_path.encode('utf-8') if self.grammar_path else None,
            grammar_string=self.grammar_string.encode('utf-8') if self.grammar_string else None,
            enable_json=ctypes.c_bool(self.enable_json),
        )


@dataclass
class GenerationConfig:
    """LLM / VLM generation configuration."""

    max_tokens: int = 512
    stop: Optional[List[str]] = None
    n_past: int = 0
    sampler_config: Optional[SamplerConfig] = None
    image_paths: Optional[List[str]] = None
    image_count: int = 0
    image_max_length: int = 0
    audio_paths: Optional[List[str]] = None
    audio_count: int = 0

    def to_c_struct(self) -> ml_GenerationConfig:
        """Convert to C structure."""
        # Convert stop sequences
        stop_ptr = None
        stop_count = 0
        if self.stop:
            stop_count = len(self.stop)
            stop_array = (ctypes.c_char_p * stop_count)()
            for i, s in enumerate(self.stop):
                stop_array[i] = s.encode('utf-8')
            stop_ptr = ctypes.cast(stop_array, ctypes.POINTER(ctypes.c_char_p))

        # Convert sampler config
        sampler_ptr = None
        if self.sampler_config:
            sampler_c = self.sampler_config.to_c_struct()
            sampler_ptr = ctypes.pointer(sampler_c)

        # Convert image paths
        image_paths_ptr = None
        image_count = 0
        if self.image_paths:
            image_count = len(self.image_paths)
            image_array = (ctypes.c_char_p * image_count)()
            for i, p in enumerate(self.image_paths):
                image_array[i] = p.encode('utf-8')
            image_paths_ptr = ctypes.cast(image_array, ctypes.POINTER(ctypes.c_char_p))
        else:
            image_count = self.image_count

        # Convert audio paths
        audio_paths_ptr = None
        audio_count = 0
        if self.audio_paths:
            audio_count = len(self.audio_paths)
            audio_array = (ctypes.c_char_p * audio_count)()
            for i, p in enumerate(self.audio_paths):
                audio_array[i] = p.encode('utf-8')
            audio_paths_ptr = ctypes.cast(audio_array, ctypes.POINTER(ctypes.c_char_p))
        else:
            audio_count = self.audio_count

        return ml_GenerationConfig(
            max_tokens=ctypes.c_int32(self.max_tokens),
            stop=stop_ptr,
            stop_count=ctypes.c_int32(stop_count),
            n_past=ctypes.c_int32(self.n_past),
            sampler_config=sampler_ptr,
            image_paths=image_paths_ptr,
            image_count=ctypes.c_int32(image_count),
            image_max_length=ctypes.c_int32(self.image_max_length),
            audio_paths=audio_paths_ptr,
            audio_count=ctypes.c_int32(audio_count),
        )


@dataclass
class ModelConfig:
    """LLM / VLM model configuration."""

    n_ctx: int = 0  # 0 = from model
    n_threads: int = 4
    n_threads_batch: int = 4
    n_batch: int = 512
    n_ubatch: int = 512
    n_seq_max: int = 1
    n_gpu_layers: int = 999  # Prefer GPU for all layers
    chat_template_path: Optional[str] = None
    chat_template_content: Optional[str] = None
    system_prompt: Optional[str] = None
    enable_sampling: bool = False  # DEPRECATED
    grammar_str: Optional[str] = None
    max_tokens: int = 512
    enable_thinking: bool = False
    verbose: bool = False
    qnn_model_folder_path: Optional[str] = None
    qnn_lib_folder_path: Optional[str] = None

    def to_c_struct(self) -> ml_ModelConfig:
        """Convert to C structure."""
        return ml_ModelConfig(
            n_ctx=ctypes.c_int32(self.n_ctx),
            n_threads=ctypes.c_int32(self.n_threads),
            n_threads_batch=ctypes.c_int32(self.n_threads_batch),
            n_batch=ctypes.c_int32(self.n_batch),
            n_ubatch=ctypes.c_int32(self.n_ubatch),
            n_seq_max=ctypes.c_int32(self.n_seq_max),
            n_gpu_layers=ctypes.c_int32(self.n_gpu_layers),
            chat_template_path=self.chat_template_path.encode('utf-8') if self.chat_template_path else None,
            chat_template_content=self.chat_template_content.encode('utf-8') if self.chat_template_content else None,
            system_prompt=self.system_prompt.encode('utf-8') if self.system_prompt else None,
            enable_sampling=ctypes.c_bool(self.enable_sampling),
            grammar_str=self.grammar_str.encode('utf-8') if self.grammar_str else None,
            max_tokens=ctypes.c_int32(self.max_tokens),
            enable_thinking=ctypes.c_bool(self.enable_thinking),
            verbose=ctypes.c_bool(self.verbose),
            qnn_model_folder_path=self.qnn_model_folder_path.encode('utf-8') if self.qnn_model_folder_path else None,
            qnn_lib_folder_path=self.qnn_lib_folder_path.encode('utf-8') if self.qnn_lib_folder_path else None,
        )


@dataclass
class LlmChatMessage:
    """Chat message structure."""

    role: str  # "user", "assistant", "system"
    content: str  # Message content in UTF-8

    def to_c_struct(self) -> ml_LlmChatMessage:
        """Convert to C structure."""
        return ml_LlmChatMessage(
            role=self.role.encode('utf-8'),
            content=self.content.encode('utf-8'),
        )


@dataclass
class VlmContent:
    """VLM content structure (text, image, audio, etc.)."""

    type: str  # "text", "image", "audio", etc.
    text: str  # Content: text content, image path, audio path, etc.

    def to_c_struct(self):
        """Convert to C structure."""
        import ctypes

        from .vlm import ml_VlmContent

        # Create string buffers to keep data alive
        type_buf = ctypes.create_string_buffer(self.type.encode('utf-8'))
        text_buf = ctypes.create_string_buffer(self.text.encode('utf-8'))

        # Store buffers as instance attributes to prevent garbage collection
        self._type_buf = type_buf
        self._text_buf = text_buf

        return ml_VlmContent(
            type=ctypes.cast(type_buf, ctypes.c_char_p),
            text=ctypes.cast(text_buf, ctypes.c_char_p),
        )


@dataclass
class VlmChatMessage:
    """VLM chat message structure."""

    role: str  # "user", "assistant", "system"
    contents: List[VlmContent]  # List of content items (text, image, etc.)

    def to_c_struct(self):
        """Convert to C structure."""
        import ctypes

        from .vlm import ml_VlmChatMessage, ml_VlmContent

        # Create string buffer for role
        role_buf = ctypes.create_string_buffer(self.role.encode('utf-8'))
        self._role_buf = role_buf

        # Convert contents to C structures
        content_count = len(self.contents)
        c_contents = (ml_VlmContent * content_count)()
        content_bufs = []

        for i, content in enumerate(self.contents):
            c_content = content.to_c_struct()
            c_contents[i] = c_content
            # Keep references to prevent garbage collection
            content_bufs.append(content._type_buf)
            content_bufs.append(content._text_buf)

        # Store all buffers as instance attributes
        self._content_bufs = content_bufs
        self._c_contents = c_contents

        return ml_VlmChatMessage(
            role=ctypes.cast(role_buf, ctypes.c_char_p),
            contents=ctypes.cast(c_contents, ctypes.POINTER(ml_VlmContent)),
            content_count=content_count,
        )


@dataclass
class ProfileData:
    """Profile data structure for performance metrics."""

    ttft: int = 0  # Time to first token (us)
    prompt_time: int = 0  # Prompt processing time (us)
    decode_time: int = 0  # Token generation time (us)
    prompt_tokens: int = 0  # Number of prompt tokens
    generated_tokens: int = 0  # Number of generated tokens
    audio_duration: int = 0  # Audio duration (us)
    prefill_speed: float = 0.0  # Prefill speed (tokens/sec)
    decoding_speed: float = 0.0  # Decoding speed (tokens/sec)
    real_time_factor: float = 0.0  # Real-Time Factor (RTF)
    stop_reason: Optional[str] = None  # Stop reason: "eos", "length", "user", "stop_sequence"

    @classmethod
    def from_c_struct(cls, c_struct: ml_ProfileData) -> 'ProfileData':
        """Create from C structure."""
        stop_reason = None
        if c_struct.stop_reason:
            stop_reason = c_struct.stop_reason.decode('utf-8')

        return cls(
            ttft=c_struct.ttft,
            prompt_time=c_struct.prompt_time,
            decode_time=c_struct.decode_time,
            prompt_tokens=c_struct.prompt_tokens,
            generated_tokens=c_struct.generated_tokens,
            audio_duration=c_struct.audio_duration,
            prefill_speed=c_struct.prefill_speed,
            decoding_speed=c_struct.decoding_speed,
            real_time_factor=c_struct.real_time_factor,
            stop_reason=stop_reason,
        )


@dataclass
class KvCacheSaveInput:
    """Input structure for saving KV cache."""

    path: str  # Path to save the KV cache

    def to_c_struct(self) -> ml_KvCacheSaveInput:
        """Convert to C structure."""
        return ml_KvCacheSaveInput(
            path=self.path.encode('utf-8'),
        )


@dataclass
class KvCacheLoadInput:
    """Input structure for loading KV cache."""

    path: str  # Path to load the KV cache from

    def to_c_struct(self) -> ml_KvCacheLoadInput:
        """Convert to C structure."""
        return ml_KvCacheLoadInput(
            path=self.path.encode('utf-8'),
        )
