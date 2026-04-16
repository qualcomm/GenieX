"""
ImageGen (Image Generation) wrapper for geniex.
Provides high-level Python API for image generation operations.
"""

import ctypes
import logging
from typing import List, Optional

from .internal.model_loader import ModelLoaderMixin
from .geniex_sdk._lib import _lib
from .geniex_sdk.error import check_error
from .geniex_sdk.image_gen import (
    ml_ImageGen,
    ml_ImageGenCreateInput,
    ml_ImageGenerationConfig,
    ml_ImageGenImg2ImgInput,
    ml_ImageGenOutput,
    ml_ImageGenTxt2ImgInput,
    ml_ImageSamplerConfig,
    ml_SchedulerConfig,
)
from .geniex_sdk.types import ModelConfig

logger = logging.getLogger(__name__)


class ImageGen(ModelLoaderMixin):
    """Image generation model wrapper."""

    def __init__(
        self,
        model_path: str,
        model_name: Optional[str] = None,
        config: Optional[ModelConfig] = None,
        scheduler_config_path: Optional[str] = None,
        plugin_id: Optional[str] = None,
        device_id: Optional[str] = None,
    ):
        """
        Create and initialize an ImageGen instance.

        Args:
            model_path: Path to the model file.
            model_name: Name of the model. If None, uses model_path.
            config: Model configuration. If None, uses default config.
            scheduler_config_path: Path to scheduler configuration file.
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
        scheduler_config_path_bytes = scheduler_config_path.encode('utf-8') if scheduler_config_path else None
        plugin_id_str = plugin_id if plugin_id else ''
        plugin_id_buf = ctypes.create_string_buffer(plugin_id_str.encode('utf-8'))
        device_id_buf = ctypes.create_string_buffer(device_id.encode('utf-8')) if device_id else None

        model_name_buf = ctypes.create_string_buffer(model_name_bytes)
        model_path_buf = ctypes.create_string_buffer(model_path_bytes)
        scheduler_config_path_buf = (
            ctypes.create_string_buffer(scheduler_config_path_bytes) if scheduler_config_path_bytes else None
        )

        c_input = ml_ImageGenCreateInput(
            model_name=ctypes.cast(model_name_buf, ctypes.c_char_p),
            model_path=ctypes.cast(model_path_buf, ctypes.c_char_p),
            config=c_config,
            scheduler_config_path=ctypes.cast(scheduler_config_path_buf, ctypes.c_char_p)
            if scheduler_config_path_buf
            else None,
            plugin_id=ctypes.cast(plugin_id_buf, ctypes.c_char_p) if plugin_id_buf else None,
            device_id=ctypes.cast(device_id_buf, ctypes.c_char_p) if device_id_buf else None,
        )

        self._string_refs = [
            model_name_buf,
            model_path_buf,
            scheduler_config_path_buf,
            plugin_id_buf,
            device_id_buf,
        ]

        handle_ptr = ctypes.POINTER(ml_ImageGen)()
        error_code = _lib.ml_imagegen_create(ctypes.pointer(c_input), ctypes.byref(handle_ptr))
        check_error(error_code)

        if not handle_ptr:
            raise RuntimeError('Failed to create ImageGen instance')

        self._handle = handle_ptr
        self._config = config

    def __del__(self):
        """Destroy ImageGen instance and free associated resources."""
        if hasattr(self, '_handle') and self._handle:
            try:
                _lib.ml_imagegen_destroy(self._handle)
            except Exception as e:
                logger.warning(f'Error during ImageGen cleanup: {e}')
            self._handle = None

    @classmethod
    def _get_model_type(cls) -> str:
        return 'image_gen'

    @classmethod
    def _extract_model_params_from_manifest(
        cls,
        manifest,
        model_path: str,
        repo_id: str,
        store,
        **kwargs,
    ) -> dict:
        """Extract ImageGen-specific parameters from manifest."""
        model_name = manifest.model_name or repo_id
        if not model_name:
            model_name = repo_id

        scheduler_config_path = None
        if hasattr(manifest, 'scheduler_config_file') and manifest.scheduler_config_file:
            scheduler_config_path_obj = store.modelfile_path(repo_id, manifest.scheduler_config_file.name)
            if scheduler_config_path_obj.exists():
                scheduler_config_path = str(scheduler_config_path_obj)

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
            'scheduler_config_path': scheduler_config_path,
            'plugin_id': plugin_id,
            'device_id': device_id,
            **kwargs,
        }

    @classmethod
    def _create_instance(cls, **params) -> 'ImageGen':
        """Create ImageGen instance with given parameters."""
        return cls(**params)

    @classmethod
    def _get_default_plugin_id(cls) -> str:
        """Get default plugin ID for ImageGen."""
        return 'cpu_gpu'

    def txt2img(
        self,
        prompt: str,
        output_path: str,
        negative_prompts: Optional[List[str]] = None,
        height: int = 512,
        width: int = 512,
        method: str = 'ddpm',
        steps: int = 50,
        guidance_scale: float = 7.5,
        eta: float = 0.0,
        seed: int = -1,
        strength: float = 1.0,
    ) -> 'ImageGenResult':
        """
        Generate image from text prompt.

        Args:
            prompt: Text prompt for image generation.
            output_path: Path to save the generated image.
            negative_prompts: List of negative prompts.
            height: Image height in pixels.
            width: Image width in pixels.
            method: Sampling method (e.g., 'ddpm', 'ddim').
            steps: Number of diffusion steps.
            guidance_scale: Guidance scale for classifier-free guidance.
            eta: Eta parameter for DDIM.
            seed: Random seed. -1 for random.
            strength: Strength parameter.

        Returns:
            ImageGenResult containing output image path.

        Raises:
            Geniex: If generation fails.
        """
        # C interface implementation
        sampler_config = ml_ImageSamplerConfig(
            method=method.encode('utf-8'),
            steps=ctypes.c_int32(steps),
            guidance_scale=ctypes.c_float(guidance_scale),
            eta=ctypes.c_float(eta),
            seed=ctypes.c_int32(seed),
        )

        scheduler_config = ml_SchedulerConfig(
            type=None,
            num_train_timesteps=0,
            steps_offset=0,
            beta_start=0.0,
            beta_end=0.0,
            beta_schedule=None,
            prediction_type=None,
            timestep_type=None,
            timestep_spacing=None,
            interpolation_type=None,
            config_path=None,
        )

        prompt_bytes = prompt.encode('utf-8')
        prompt_buf = ctypes.create_string_buffer(prompt_bytes)
        prompts_array = (ctypes.c_char_p * 1)(ctypes.cast(prompt_buf, ctypes.c_char_p))

        negative_prompts_ptr = None
        negative_prompt_count = 0
        if negative_prompts:
            negative_prompt_count = len(negative_prompts)
            negative_prompts_array = (ctypes.c_char_p * negative_prompt_count)()
            negative_prompt_bufs = []
            for i, neg_prompt in enumerate(negative_prompts):
                neg_prompt_bytes = neg_prompt.encode('utf-8')
                neg_prompt_buf = ctypes.create_string_buffer(neg_prompt_bytes)
                negative_prompt_bufs.append(neg_prompt_buf)
                negative_prompts_array[i] = ctypes.cast(neg_prompt_buf, ctypes.c_char_p)
            negative_prompts_ptr = ctypes.cast(negative_prompts_array, ctypes.POINTER(ctypes.c_char_p))
            self._negative_prompt_bufs = negative_prompt_bufs

        generation_config = ml_ImageGenerationConfig(
            prompts=ctypes.cast(prompts_array, ctypes.POINTER(ctypes.c_char_p)),
            prompt_count=ctypes.c_int32(1),
            negative_prompts=negative_prompts_ptr,
            negative_prompt_count=ctypes.c_int32(negative_prompt_count),
            height=ctypes.c_int32(height),
            width=ctypes.c_int32(width),
            sampler_config=sampler_config,
            scheduler_config=scheduler_config,
            strength=ctypes.c_float(strength),
        )

        prompt_utf8_bytes = prompt.encode('utf-8')
        prompt_utf8_buf = ctypes.create_string_buffer(prompt_utf8_bytes)
        output_path_bytes = output_path.encode('utf-8')
        output_path_buf = ctypes.create_string_buffer(output_path_bytes)

        c_input = ml_ImageGenTxt2ImgInput(
            prompt_utf8=ctypes.cast(prompt_utf8_buf, ctypes.c_char_p),
            config=ctypes.pointer(generation_config),
            output_path=ctypes.cast(output_path_buf, ctypes.c_char_p),
        )

        c_output = ml_ImageGenOutput()
        error_code = _lib.ml_imagegen_txt2img(self._handle, ctypes.pointer(c_input), ctypes.pointer(c_output))
        check_error(error_code)

        return self._extract_result(c_output)

    def img2img(
        self,
        init_image_path: str,
        prompt: str,
        output_path: str,
        negative_prompts: Optional[List[str]] = None,
        height: int = 512,
        width: int = 512,
        method: str = 'ddpm',
        steps: int = 50,
        guidance_scale: float = 7.5,
        eta: float = 0.0,
        seed: int = -1,
        strength: float = 0.8,
    ) -> 'ImageGenResult':
        """
        Generate image from existing image and text prompt.

        Args:
            init_image_path: Path to the initial image.
            prompt: Text prompt for image generation.
            output_path: Path to save the generated image.
            negative_prompts: List of negative prompts.
            height: Image height in pixels.
            width: Image width in pixels.
            method: Sampling method (e.g., 'ddpm', 'ddim').
            steps: Number of diffusion steps.
            guidance_scale: Guidance scale for classifier-free guidance.
            eta: Eta parameter for DDIM.
            seed: Random seed. -1 for random.
            strength: Strength parameter (0.0-1.0).

        Returns:
            ImageGenResult containing output image path.

        Raises:
            Geniex: If generation fails.
        """
        # C interface implementation
        sampler_config = ml_ImageSamplerConfig(
            method=method.encode('utf-8'),
            steps=ctypes.c_int32(steps),
            guidance_scale=ctypes.c_float(guidance_scale),
            eta=ctypes.c_float(eta),
            seed=ctypes.c_int32(seed),
        )

        scheduler_config = ml_SchedulerConfig(
            type=None,
            num_train_timesteps=0,
            steps_offset=0,
            beta_start=0.0,
            beta_end=0.0,
            beta_schedule=None,
            prediction_type=None,
            timestep_type=None,
            timestep_spacing=None,
            interpolation_type=None,
            config_path=None,
        )

        prompt_bytes = prompt.encode('utf-8')
        prompt_buf = ctypes.create_string_buffer(prompt_bytes)
        prompts_array = (ctypes.c_char_p * 1)(ctypes.cast(prompt_buf, ctypes.c_char_p))

        negative_prompts_ptr = None
        negative_prompt_count = 0
        if negative_prompts:
            negative_prompt_count = len(negative_prompts)
            negative_prompts_array = (ctypes.c_char_p * negative_prompt_count)()
            negative_prompt_bufs = []
            for i, neg_prompt in enumerate(negative_prompts):
                neg_prompt_bytes = neg_prompt.encode('utf-8')
                neg_prompt_buf = ctypes.create_string_buffer(neg_prompt_bytes)
                negative_prompt_bufs.append(neg_prompt_buf)
                negative_prompts_array[i] = ctypes.cast(neg_prompt_buf, ctypes.c_char_p)
            negative_prompts_ptr = ctypes.cast(negative_prompts_array, ctypes.POINTER(ctypes.c_char_p))
            self._negative_prompt_bufs = negative_prompt_bufs

        generation_config = ml_ImageGenerationConfig(
            prompts=ctypes.cast(prompts_array, ctypes.POINTER(ctypes.c_char_p)),
            prompt_count=ctypes.c_int32(1),
            negative_prompts=negative_prompts_ptr,
            negative_prompt_count=ctypes.c_int32(negative_prompt_count),
            height=ctypes.c_int32(height),
            width=ctypes.c_int32(width),
            sampler_config=sampler_config,
            scheduler_config=scheduler_config,
            strength=ctypes.c_float(strength),
        )

        init_image_path_bytes = init_image_path.encode('utf-8')
        init_image_path_buf = ctypes.create_string_buffer(init_image_path_bytes)
        prompt_utf8_bytes = prompt.encode('utf-8')
        prompt_utf8_buf = ctypes.create_string_buffer(prompt_utf8_bytes)
        output_path_bytes = output_path.encode('utf-8')
        output_path_buf = ctypes.create_string_buffer(output_path_bytes)

        c_input = ml_ImageGenImg2ImgInput(
            init_image_path=ctypes.cast(init_image_path_buf, ctypes.c_char_p),
            prompt_utf8=ctypes.cast(prompt_utf8_buf, ctypes.c_char_p),
            config=ctypes.pointer(generation_config),
            output_path=ctypes.cast(output_path_buf, ctypes.c_char_p),
        )

        c_output = ml_ImageGenOutput()
        error_code = _lib.ml_imagegen_img2img(self._handle, ctypes.pointer(c_input), ctypes.pointer(c_output))
        check_error(error_code)

        return self._extract_result(c_output)

    def _extract_result(self, c_output: ml_ImageGenOutput) -> 'ImageGenResult':
        """Extract ImageGenResult from C output structure."""
        output_image_path = ''
        if c_output.output_image_path:
            output_image_path = c_output.output_image_path.decode('utf-8')

        return ImageGenResult(output_image_path=output_image_path)


class ImageGenResult:
    """Result of image generation."""

    def __init__(self, output_image_path: str):
        self.output_image_path = output_image_path

    def __repr__(self) -> str:
        return f'ImageGenResult(output_image_path={self.output_image_path})'
