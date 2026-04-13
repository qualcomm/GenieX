"""
VLM (Vision-Language Model) wrapper for geniex.
Provides high-level Python API for VLM operations.
"""

import ctypes
import logging
import queue
import threading
from typing import Callable, Generator, List, Optional

from .internal.model_loader import ModelLoaderMixin
from .geniex_sdk._lib import _lib
from .geniex_sdk.error import check_error
from .geniex_sdk.llm import ml_token_callback
from .geniex_sdk.types import GenerationConfig, ModelConfig, ProfileData, VlmChatMessage
from .geniex_sdk.vlm import (
    ml_VLM,
    ml_VlmApplyChatTemplateInput,
    ml_VlmApplyChatTemplateOutput,
    ml_VlmChatMessage,
    ml_VlmCreateInput,
    ml_VlmGenerateInput,
    ml_VlmGenerateOutput,
)

logger = logging.getLogger(__name__)


class VLM(ModelLoaderMixin):
    """Vision-Language Model wrapper."""

    def __init__(
        self,
        model_path: str,
        mmproj_path: Optional[str] = None,
        tokenizer_path: Optional[str] = None,
        model_name: Optional[str] = None,
        config: Optional[ModelConfig] = None,
        plugin_id: Optional[str] = None,
        device_id: Optional[str] = None,
        license_id: Optional[str] = None,
        license_key: Optional[str] = None,
    ):
        """
        Create and initialize a VLM instance.

        Args:
            model_path: Path to the model file.
            mmproj_path: Path to the mmproj file. If None, uses model_path.
            tokenizer_path: Path to the tokenizer file. If None, uses model_path.
            model_name: Name of the model. If None, uses model_path.
            config: Model configuration. If None, uses default config.
            plugin_id: Plugin to use for the model. If None, uses default.
            device_id: Device to use for the model. If None, uses default device.
            license_id: License ID for loading NPU models.
            license_key: License key for loading NPU models.
        """
        if config is None:
            config = ModelConfig()

        if mmproj_path is None:
            mmproj_path = model_path

        if tokenizer_path is None:
            tokenizer_path = model_path

        if model_name is None:
            model_name = model_path

        # Ensure model_name is not empty
        if not model_name:
            model_name = model_path

        # C interface initialization
        c_config = config.to_c_struct()

        # Create string buffers
        model_name_bytes = model_name.encode('utf-8')
        model_path_bytes = model_path.encode('utf-8')
        mmproj_path_bytes = mmproj_path.encode('utf-8')
        tokenizer_path_bytes = tokenizer_path.encode('utf-8') if tokenizer_path else b''

        model_name_buf = ctypes.create_string_buffer(model_name_bytes)
        model_path_buf = ctypes.create_string_buffer(model_path_bytes)
        mmproj_path_buf = ctypes.create_string_buffer(mmproj_path_bytes)
        tokenizer_path_buf = ctypes.create_string_buffer(tokenizer_path_bytes) if tokenizer_path_bytes else None
        plugin_id_buf = ctypes.create_string_buffer(plugin_id.encode('utf-8')) if plugin_id else None
        device_id_buf = ctypes.create_string_buffer(device_id.encode('utf-8')) if device_id else None
        license_id_buf = ctypes.create_string_buffer(license_id.encode('utf-8')) if license_id else None
        license_key_buf = ctypes.create_string_buffer(license_key.encode('utf-8')) if license_key else None

        # Create C input structure
        c_input = ml_VlmCreateInput(
            model_name=ctypes.cast(model_name_buf, ctypes.c_char_p),
            model_path=ctypes.cast(model_path_buf, ctypes.c_char_p),
            mmproj_path=ctypes.cast(mmproj_path_buf, ctypes.c_char_p),
            tokenizer_path=ctypes.cast(tokenizer_path_buf, ctypes.c_char_p) if tokenizer_path_buf else None,
            config=c_config,
            plugin_id=ctypes.cast(plugin_id_buf, ctypes.c_char_p) if plugin_id_buf else None,
            device_id=ctypes.cast(device_id_buf, ctypes.c_char_p) if device_id_buf else None,
            license_id=ctypes.cast(license_id_buf, ctypes.c_char_p) if license_id_buf else None,
            license_key=ctypes.cast(license_key_buf, ctypes.c_char_p) if license_key_buf else None,
        )

        # Keep references to prevent garbage collection
        self._string_refs = [
            model_name_buf,
            model_path_buf,
            mmproj_path_buf,
            tokenizer_path_buf,
            plugin_id_buf,
            device_id_buf,
            license_id_buf,
            license_key_buf,
        ]

        # Create VLM instance
        handle_ptr = ctypes.POINTER(ml_VLM)()
        error_code = _lib.ml_vlm_create(ctypes.pointer(c_input), ctypes.byref(handle_ptr))
        check_error(error_code)

        if not handle_ptr:
            raise RuntimeError('Failed to create VLM instance')

        self._handle = handle_ptr
        self._config = config

    def __del__(self):
        """Destroy VLM instance and free associated resources."""
        if hasattr(self, '_handle') and self._handle:
            try:
                _lib.ml_vlm_destroy(self._handle)
            except Exception as e:
                logger.warning(f'Error during VLM cleanup: {e}')
                pass
            self._handle = None

    @classmethod
    def _get_model_type(cls) -> str:
        return 'vlm'

    @classmethod
    def _extract_model_params_from_manifest(
        cls,
        manifest,
        model_path: str,
        repo_id: str,
        store,
        **kwargs,
    ) -> dict:
        """
        Extract VLM-specific parameters from manifest.

        Args:
            manifest: Model manifest
            model_path: Path to the model file
            repo_id: Repository ID
            store: Store instance
            kwargs: Additional arguments

        Returns:
            Dictionary of parameters for VLM.__init__()
        """
        # Get mmproj path if available
        mmproj_path = None
        if manifest.mmproj_file.name:
            mmproj_path_obj = store.modelfile_path(repo_id, manifest.mmproj_file.name)
            if mmproj_path_obj.exists():
                mmproj_path = str(mmproj_path_obj)

        # Get tokenizer path if available
        tokenizer_path = None
        if manifest.tokenizer_file.name:
            tokenizer_path_obj = store.modelfile_path(repo_id, manifest.tokenizer_file.name)
            if tokenizer_path_obj.exists():
                tokenizer_path = str(tokenizer_path_obj)

        model_name = kwargs.pop('model_name', None) or manifest.model_name or repo_id
        plugin_id = kwargs.pop('plugin_id', None) or manifest.plugin_id or cls._get_default_plugin_id()
        device_id = kwargs.pop('device_id', None) or manifest.device_id or None

        return {
            'model_path': str(model_path),
            'mmproj_path': mmproj_path,
            'tokenizer_path': tokenizer_path,
            'model_name': model_name,
            'plugin_id': plugin_id,
            'device_id': device_id,
            **kwargs,
        }

    @classmethod
    def _create_instance(cls, **params) -> 'VLM':
        """
        Create VLM instance with given parameters.

        Args:
            params: Parameters for VLM.__init__()

        Returns:
            VLM instance
        """
        return cls(**params)

    def reset(self) -> None:
        """
        Reset VLM internal state.

        Raises:
            Geniex: If reset fails.
        """
        error_code = _lib.ml_vlm_reset(self._handle)
        check_error(error_code)

    def _generate_impl(
        self,
        prompt: str,
        config: Optional[GenerationConfig],
        token_callback: Optional[Callable[[ctypes.c_char_p, ctypes.c_void_p], bool]],
    ) -> tuple[ml_VlmGenerateOutput, ctypes.c_void_p]:
        """Internal implementation for text generation."""
        # Convert generation config to C struct
        c_gen_config = None
        if config:
            c_gen_config = config.to_c_struct()

        # Create token callback wrapper
        if token_callback:
            token_callback_wrapper = ml_token_callback(token_callback)
        else:

            def _dummy_callback(token_ptr: ctypes.c_char_p, user_data_ptr: ctypes.c_void_p) -> bool:
                return True

            token_callback_wrapper = ml_token_callback(_dummy_callback)

        # Create string buffer to keep prompt data alive during the call
        prompt_bytes = prompt.encode('utf-8')
        prompt_buf = ctypes.create_string_buffer(prompt_bytes)

        # Create C input structure
        c_input = ml_VlmGenerateInput(
            prompt_utf8=ctypes.cast(prompt_buf, ctypes.c_char_p),
            config=ctypes.pointer(c_gen_config) if c_gen_config else None,
            on_token=token_callback_wrapper,
            user_data=None,
        )

        # Keep references alive during the call
        _callback_ref = token_callback_wrapper
        _prompt_ref = prompt_buf

        c_output = ml_VlmGenerateOutput()
        error_code = _lib.ml_vlm_generate(self._handle, ctypes.pointer(c_input), ctypes.pointer(c_output))
        check_error(error_code)

        return c_output, _callback_ref

    def generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
        on_token: Optional[Callable[[str], bool]] = None,
    ) -> 'GenerateResult':
        """
        Generate text with optional streaming token callback.

        Args:
            prompt: The prompt text.
            config: Generation configuration. If None, uses default config.
            on_token: Optional callback function for streaming tokens.
                     Should return True to continue, False to stop.

        Returns:
            GenerateResult containing the generated text and profile data.

        Raises:
            Geniex: If generation fails.
        """
        token_callback = None
        if on_token:

            def _token_callback(token_ptr: ctypes.c_char_p, user_data_ptr: ctypes.c_void_p) -> bool:
                if token_ptr:
                    token = token_ptr.decode('utf-8')
                    return on_token(token)
                return True

            token_callback = _token_callback

        c_output, _ = self._generate_impl(prompt, config, token_callback)
        return self._extract_result(c_output)

    def generate_stream(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
    ) -> Generator[str, None, 'GenerateResult']:
        """
        Stream text generation token by token.

        Args:
            prompt: The prompt text.
            config: Generation configuration. If None, uses default config.

        Yields:
            str: Each generated token as it becomes available.

        Returns:
            GenerateResult containing the full generated text and profile data.

        Raises:
            Geniex: If generation fails.

        Example:
            >>> vlm = VLM(model_path="path/to/model")
            >>> result = None
            >>> for token in vlm.generate_stream("Describe the image."):
            ...     print(token, end='', flush=True)
            >>> print(f"\\nFull text: {result.full_text}")
        """
        token_queue: queue.Queue[Optional[str]] = queue.Queue()
        generation_done = threading.Event()
        c_output_ref = [None]
        error_ref = [None]

        def _token_callback(token_ptr: ctypes.c_char_p, user_data_ptr: ctypes.c_void_p) -> bool:
            if token_ptr:
                token = token_ptr.decode('utf-8')
                token_queue.put(token)
            return True

        def _generate_in_thread():
            try:
                c_output, _ = self._generate_impl(prompt, config, _token_callback)
                c_output_ref[0] = c_output
            except Exception as e:
                error_ref[0] = e
            finally:
                token_queue.put(None)  # Signal end of generation
                generation_done.set()

        # Start generation in a separate thread
        gen_thread = threading.Thread(target=_generate_in_thread, daemon=True)
        gen_thread.start()

        # Yield tokens as they arrive (blocking wait for first token, then non-blocking)
        while True:
            try:
                # Use blocking get with timeout to ensure we yield tokens as soon as they arrive
                token = token_queue.get(timeout=1.0)
                if token is None:  # End of generation
                    break
                yield token
            except queue.Empty:
                # Check if generation is done
                if generation_done.is_set():
                    # Try to get any remaining tokens
                    try:
                        while True:
                            token = token_queue.get_nowait()
                            if token is None:
                                break
                            yield token
                    except queue.Empty:
                        pass
                    break
                continue

        # Wait for thread to complete
        gen_thread.join()

        # Check for errors
        if error_ref[0]:
            raise error_ref[0]

        # Return the final result
        return self._extract_result(c_output_ref[0])

    def _extract_result(self, c_output: ml_VlmGenerateOutput) -> 'GenerateResult':
        """Extract GenerateResult from C output structure."""
        full_text = ''
        if c_output.full_text:
            full_text = ctypes.cast(c_output.full_text, ctypes.c_char_p).value.decode('utf-8')
            _lib.ml_free(c_output.full_text)
        profile_data = ProfileData.from_c_struct(c_output.profile_data)
        return GenerateResult(full_text=full_text, profile_data=profile_data)

    def apply_chat_template(
        self,
        messages: List[VlmChatMessage],
        tools: Optional[str] = None,
        enable_thinking: bool = False,
    ) -> str:
        """
        Apply chat template to messages.

        Args:
            messages: Array of chat messages.
            tools: Optional tool JSON string.
            enable_thinking: Enable thinking mode.

        Returns:
            Formatted chat text.

        Raises:
            Geniex: If template application fails.
        """
        # Convert messages to C structures
        message_count = len(messages)
        c_messages = (ml_VlmChatMessage * message_count)()

        # Keep all references to prevent garbage collection during C call
        _refs = []  # Keep all buffers and messages alive

        for i, msg in enumerate(messages):
            if isinstance(msg, VlmChatMessage):
                c_msg = msg.to_c_struct()
                c_messages[i] = c_msg
                _refs.append(msg)
                if hasattr(msg, '_role_buf'):
                    _refs.append(msg._role_buf)
                if hasattr(msg, '_c_contents'):
                    _refs.append(msg._c_contents)
                if hasattr(msg, '_content_bufs'):
                    _refs.extend(msg._content_bufs)
                for content in msg.contents:
                    if hasattr(content, '_type_buf'):
                        _refs.append(content._type_buf)
                    if hasattr(content, '_text_buf'):
                        _refs.append(content._text_buf)
            elif isinstance(msg, ml_VlmChatMessage):
                c_messages[i] = msg
            else:
                raise TypeError(f'Expected VlmChatMessage or ml_VlmChatMessage, got {type(msg)}')

        # Create tools buffer if needed
        tools_buf = None
        if tools:
            tools_buf = ctypes.create_string_buffer(tools.encode('utf-8'))
            _refs.append(tools_buf)
            tools_ptr = ctypes.cast(tools_buf, ctypes.c_char_p)
        else:
            tools_ptr = None

        # Create C input structure
        c_input = ml_VlmApplyChatTemplateInput(
            messages=ctypes.cast(c_messages, ctypes.POINTER(ml_VlmChatMessage)),
            message_count=ctypes.c_int32(message_count),
            tools=tools_ptr,
            enable_thinking=ctypes.c_bool(enable_thinking),
            grounding=ctypes.c_bool(False),
        )

        # Create C output structure
        c_output = ml_VlmApplyChatTemplateOutput()

        error_code = _lib.ml_vlm_apply_chat_template(
            self._handle,
            ctypes.pointer(c_input),
            ctypes.pointer(c_output),
        )
        check_error(error_code)

        # Extract result
        formatted_text = ''
        if c_output.formatted_text:
            formatted_text = ctypes.cast(c_output.formatted_text, ctypes.c_char_p).value.decode('utf-8')
            # formatted_text is c_void_p, can pass directly to ml_free
            _lib.ml_free(c_output.formatted_text)

        return formatted_text


class GenerateResult:
    """Result of text generation."""

    def __init__(self, full_text: str, profile_data: ProfileData):
        self.full_text = full_text
        self.profile_data = profile_data

    @property
    def text(self) -> str:
        """Alias for full_text for compatibility."""
        return self.full_text

    def __str__(self) -> str:
        return self.full_text

    def __repr__(self) -> str:
        return f'GenerateResult(full_text={self.full_text[:50]}..., profile_data={self.profile_data})'
