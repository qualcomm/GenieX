"""
LLM (Large Language Model) wrapper for geniex.
Provides high-level Python API for LLM operations.
"""

import ctypes
import logging
import queue
import threading
from typing import Callable, Generator, List, Optional

from .internal.model_loader import ModelLoaderMixin
from .geniex_sdk._lib import _lib
from .geniex_sdk.error import check_error
from .geniex_sdk.llm import (
    ml_KvCacheLoadOutput,
    ml_KvCacheSaveOutput,
    ml_LLM,
    ml_LlmApplyChatTemplateInput,
    ml_LlmApplyChatTemplateOutput,
    ml_LlmChatMessage,
    ml_LlmCreateInput,
    ml_LlmGenerateInput,
    ml_LlmGenerateOutput,
    ml_token_callback,
)
from .geniex_sdk.types import (
    GenerationConfig,
    KvCacheLoadInput,
    KvCacheSaveInput,
    LlmChatMessage,
    ModelConfig,
    ProfileData,
)

logger = logging.getLogger(__name__)


class LLM(ModelLoaderMixin):
    """Large Language Model wrapper."""

    def __init__(
        self,
        model_path: str,
        tokenizer_path: Optional[str] = None,
        model_name: Optional[str] = None,
        config: Optional[ModelConfig] = None,
        plugin_id: Optional[str] = None,
        device_id: Optional[str] = None,
        license_id: Optional[str] = None,
        license_key: Optional[str] = None,
    ):
        """
        Create and initialize an LLM instance.

        Args:
            model_path: Path to the model file.
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

        if tokenizer_path is None:
            tokenizer_path = model_path

        if model_name is None:
            model_name = model_path

        # Ensure model_name is not empty (C++ code may require non-NULL)
        if not model_name:
            model_name = model_path

        c_config = config.to_c_struct()

        # Create string buffers to ensure strings remain valid in memory
        # ctypes requires that string data remains valid during the function call
        # model_name must be non-NULL based on the error
        model_name_bytes = model_name.encode('utf-8')
        model_path_bytes = model_path.encode('utf-8')
        tokenizer_path_bytes = tokenizer_path.encode('utf-8') if tokenizer_path else b''

        model_name_buf = ctypes.create_string_buffer(model_name_bytes)
        model_path_buf = ctypes.create_string_buffer(model_path_bytes)
        tokenizer_path_buf = ctypes.create_string_buffer(tokenizer_path_bytes) if tokenizer_path_bytes else None
        plugin_id_str = plugin_id if plugin_id else ''
        plugin_id_buf = ctypes.create_string_buffer(plugin_id_str.encode('utf-8'))
        device_id_buf = ctypes.create_string_buffer(device_id.encode('utf-8')) if device_id else None
        license_id_buf = ctypes.create_string_buffer(license_id.encode('utf-8')) if license_id else None
        license_key_buf = ctypes.create_string_buffer(license_key.encode('utf-8')) if license_key else None

        # Create C input structure
        c_input = ml_LlmCreateInput(
            model_name=ctypes.cast(model_name_buf, ctypes.c_char_p),
            model_path=ctypes.cast(model_path_buf, ctypes.c_char_p),
            tokenizer_path=ctypes.cast(tokenizer_path_buf, ctypes.c_char_p) if tokenizer_path_buf else None,
            config=c_config,
            plugin_id=ctypes.cast(plugin_id_buf, ctypes.c_char_p) if plugin_id_buf else None,
            device_id=ctypes.cast(device_id_buf, ctypes.c_char_p) if device_id_buf else None,
            license_id=ctypes.cast(license_id_buf, ctypes.c_char_p) if license_id_buf else None,
            license_key=ctypes.cast(license_key_buf, ctypes.c_char_p) if license_key_buf else None,
        )

        # Keep references to prevent garbage collection during the call
        self._string_refs = [
            model_name_buf,
            model_path_buf,
            tokenizer_path_buf,
            plugin_id_buf,
            device_id_buf,
            license_id_buf,
            license_key_buf,
        ]

        # Create LLM instance
        handle_ptr = ctypes.POINTER(ml_LLM)()
        error_code = _lib.ml_llm_create(ctypes.pointer(c_input), ctypes.byref(handle_ptr))
        check_error(error_code)

        if not handle_ptr:
            raise RuntimeError('Failed to create LLM instance')

        self._handle = handle_ptr
        self._config = config

    def __del__(self):
        """Destroy LLM instance and free associated resources."""
        if hasattr(self, '_handle') and self._handle:
            try:
                _lib.ml_llm_destroy(self._handle)
            except Exception as e:
                logger.warning(f'Error during LLM cleanup: {e}')
                pass  # Ignore errors during cleanup
            self._handle = None

    @classmethod
    def _get_model_type(cls) -> str:
        return 'llm'

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
        Extract LLM-specific parameters from manifest.

        Args:
            manifest: Model manifest
            model_path: Path to the model file
            repo_id: Repository ID
            store: Store instance
            kwargs: Additional arguments

        Returns:
            Dictionary of parameters for LLM.__init__()
        """

        # Get tokenizer path if available
        tokenizer_path = None
        if manifest.tokenizer_file.name:
            tokenizer_path_obj = store.modelfile_path(repo_id, manifest.tokenizer_file.name)
            if tokenizer_path_obj.exists():
                tokenizer_path = str(tokenizer_path_obj)

        # Extract model_name from manifest or use repo_id
        model_name = manifest.model_name or repo_id
        if not model_name:
            model_name = repo_id

        # Get plugin_id and device_id, ensuring empty strings become None
        plugin_id = manifest.plugin_id.strip() if manifest.plugin_id else None
        if not plugin_id:
            plugin_id = kwargs.pop('plugin_id', None)
        else:
            # Remove plugin_id from kwargs if manifest has it (to avoid passing it twice)
            kwargs.pop('plugin_id', None)
            logger.debug(f'Using plugin_id from manifest: {plugin_id}')
        if plugin_id:
            plugin_id = plugin_id.strip() if isinstance(plugin_id, str) else plugin_id
            if not plugin_id:
                plugin_id = None

        # If still None, use default
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
            'tokenizer_path': tokenizer_path,
            'model_name': model_name,
            'plugin_id': plugin_id,
            'device_id': device_id,
            **kwargs,
        }

    @classmethod
    def _create_instance(cls, **params) -> 'LLM':
        """
        Create LLM instance with given parameters.

        Args:
            params: Parameters for LLM.__init__()

        Returns:
            LLM instance
        """
        return cls(**params)

    def reset(self) -> None:
        """
        Reset LLM internal state (clear KV cache, reset sampling).

        Raises:
            Geniex: If reset fails.
        """
        error_code = _lib.ml_llm_reset(self._handle)
        check_error(error_code)

    def _generate_impl(
        self,
        prompt: str,
        config: Optional[GenerationConfig],
        token_callback: Optional[Callable[[ctypes.c_char_p, ctypes.c_void_p], bool]],
    ) -> tuple[ml_LlmGenerateOutput, ctypes.c_void_p]:
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

        # Create C input structure
        c_input = ml_LlmGenerateInput(
            prompt_utf8=prompt.encode('utf-8'),
            config=ctypes.pointer(c_gen_config) if c_gen_config else None,
            on_token=token_callback_wrapper,
            user_data=None,
        )

        # Keep callback reference alive during the call
        _callback_ref = token_callback_wrapper

        c_output = ml_LlmGenerateOutput()
        error_code = _lib.ml_llm_generate(self._handle, ctypes.pointer(c_input), ctypes.pointer(c_output))
        check_error(error_code)

        return c_output, _callback_ref

    def _extract_result(self, c_output: ml_LlmGenerateOutput) -> 'GenerateResult':
        """Extract GenerateResult from C output structure."""
        full_text = ''
        if c_output.full_text:
            full_text = ctypes.cast(c_output.full_text, ctypes.c_char_p).value.decode('utf-8')
            _lib.ml_free(c_output.full_text)
        profile_data = ProfileData.from_c_struct(c_output.profile_data)
        return GenerateResult(full_text=full_text, profile_data=profile_data)

    def generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
        on_token: Optional[Callable[[str], bool]] = None,
    ) -> 'GenerateResult':
        """
        Generate text with optional streaming token callback.

        Args:
            prompt: The full chat history as UTF-8 string.
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
            prompt: The full chat history as UTF-8 string.
            config: Generation configuration. If None, uses default config.

        Yields:
            str: Each generated token as it becomes available.

        Returns:
            GenerateResult containing the full generated text and profile data.

        Raises:
            Geniex: If generation fails.

        Example:
            >>> llm = LLM(model_path="path/to/model")
            >>> result = None
            >>> for token in llm.generate_stream("Hello, world!"):
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

    def apply_chat_template(
        self,
        messages: List[LlmChatMessage],
        tools: Optional[str] = None,
        enable_thinking: bool = False,
        add_generation_prompt: bool = True,
    ) -> str:
        """
        Apply chat template to messages.

        Args:
            messages: Array of chat messages.
            tools: Optional tool JSON string.
            enable_thinking: Enable thinking mode.
            add_generation_prompt: Add generation prompt.

        Returns:
            Formatted chat text.

        Raises:
            Geniex: If template application fails.
        """
        # Convert messages to C structures
        message_count = len(messages)
        c_messages = (ml_LlmChatMessage * message_count)()
        for i, msg in enumerate(messages):
            c_messages[i] = msg.to_c_struct()

        # Create C input structure
        c_input = ml_LlmApplyChatTemplateInput(
            messages=ctypes.cast(c_messages, ctypes.POINTER(ml_LlmChatMessage)),
            message_count=ctypes.c_int32(message_count),
            tools=tools.encode('utf-8') if tools else None,
            enable_thinking=ctypes.c_bool(enable_thinking),
            add_generation_prompt=ctypes.c_bool(add_generation_prompt),
        )

        # Create C output structure
        c_output = ml_LlmApplyChatTemplateOutput()

        # Apply chat template
        error_code = _lib.ml_llm_apply_chat_template(
            self._handle,
            ctypes.pointer(c_input),
            ctypes.pointer(c_output),
        )
        check_error(error_code)

        # Extract result
        formatted_text = ''
        if c_output.formatted_text:
            formatted_text = ctypes.cast(c_output.formatted_text, ctypes.c_char_p).value.decode('utf-8')
            _lib.ml_free(c_output.formatted_text)

        return formatted_text

    def save_kv_cache(self, path: str) -> None:
        """
        Save current KV cache state to file.

        Args:
            path: Path to save the KV cache.

        Raises:
            Geniex: If save fails.
        """
        input_obj = KvCacheSaveInput(path=path)
        c_input = input_obj.to_c_struct()
        c_output = ml_KvCacheSaveOutput()

        error_code = _lib.ml_llm_save_kv_cache(
            self._handle,
            ctypes.pointer(c_input),
            ctypes.pointer(c_output),
        )
        check_error(error_code)

    def load_kv_cache(self, path: str) -> None:
        """
        Load KV cache state from file.

        Args:
            path: Path to load the KV cache from.

        Raises:
            Geniex: If load fails.
        """
        input_obj = KvCacheLoadInput(path=path)
        c_input = input_obj.to_c_struct()
        c_output = ml_KvCacheLoadOutput()

        error_code = _lib.ml_llm_load_kv_cache(
            self._handle,
            ctypes.pointer(c_input),
            ctypes.pointer(c_output),
        )
        check_error(error_code)


class GenerateResult:
    """Result of text generation."""

    def __init__(self, full_text: str, profile_data: ProfileData):
        self.full_text = full_text
        self.profile_data = profile_data

    def __str__(self) -> str:
        return self.full_text

    def __repr__(self) -> str:
        return f'GenerateResult(full_text={self.full_text[:50]}..., profile_data={self.profile_data})'
