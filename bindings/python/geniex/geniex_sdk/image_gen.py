"""
Image generation functions and structures for nexa-sdk.
"""

import ctypes

from ._lib import _lib
from .error import ml_log_callback
from .llm import ml_ModelConfig

# ============================================================================
# C Structure Definitions
# ============================================================================


class ml_ImageSamplerConfig(ctypes.Structure):
    """Image generation sampling parameters."""

    _fields_ = [
        ('method', ctypes.c_char_p),
        ('steps', ctypes.c_int32),
        ('guidance_scale', ctypes.c_float),
        ('eta', ctypes.c_float),
        ('seed', ctypes.c_int32),
    ]


class ml_SchedulerConfig(ctypes.Structure):
    """Diffusion scheduler configuration."""

    _fields_ = [
        ('type', ctypes.c_char_p),
        ('num_train_timesteps', ctypes.c_int32),
        ('steps_offset', ctypes.c_int32),
        ('beta_start', ctypes.c_float),
        ('beta_end', ctypes.c_float),
        ('beta_schedule', ctypes.c_char_p),
        ('prediction_type', ctypes.c_char_p),
        ('timestep_type', ctypes.c_char_p),
        ('timestep_spacing', ctypes.c_char_p),
        ('interpolation_type', ctypes.c_char_p),
        ('config_path', ctypes.c_char_p),
    ]


class ml_ImageGenerationConfig(ctypes.Structure):
    """Image generation configuration."""

    _fields_ = [
        ('prompts', ctypes.POINTER(ctypes.c_char_p)),
        ('prompt_count', ctypes.c_int32),
        ('negative_prompts', ctypes.POINTER(ctypes.c_char_p)),
        ('negative_prompt_count', ctypes.c_int32),
        ('height', ctypes.c_int32),
        ('width', ctypes.c_int32),
        ('sampler_config', ml_ImageSamplerConfig),
        ('scheduler_config', ml_SchedulerConfig),
        ('strength', ctypes.c_float),
    ]


class ml_ImageGen(ctypes.Structure):
    """Opaque image generator handle."""

    pass


class ml_ImageGenCreateInput(ctypes.Structure):
    """Input structure for creating an image generator."""

    _fields_ = [
        ('model_name', ctypes.c_char_p),
        ('model_path', ctypes.c_char_p),
        ('config', ml_ModelConfig),
        ('scheduler_config_path', ctypes.c_char_p),
        ('plugin_id', ctypes.c_char_p),
        ('device_id', ctypes.c_char_p),
    ]


class ml_ImageGenTxt2ImgInput(ctypes.Structure):
    """Input structure for text-to-image generation."""

    _fields_ = [
        ('prompt_utf8', ctypes.c_char_p),
        ('config', ctypes.POINTER(ml_ImageGenerationConfig)),
        ('output_path', ctypes.c_char_p),
    ]


class ml_ImageGenImg2ImgInput(ctypes.Structure):
    """Input structure for image-to-image generation."""

    _fields_ = [
        ('init_image_path', ctypes.c_char_p),
        ('prompt_utf8', ctypes.c_char_p),
        ('config', ctypes.POINTER(ml_ImageGenerationConfig)),
        ('output_path', ctypes.c_char_p),
    ]


class ml_ImageGenOutput(ctypes.Structure):
    """Output structure for image generation."""

    _fields_ = [
        ('output_image_path', ctypes.c_char_p),
    ]


# ============================================================================
# Function Signatures
# ============================================================================

_lib.ml_imagegen_create.argtypes = [
    ctypes.POINTER(ml_ImageGenCreateInput),
    ctypes.POINTER(ctypes.POINTER(ml_ImageGen)),
]
_lib.ml_imagegen_create.restype = ctypes.c_int32

_lib.ml_imagegen_destroy.argtypes = [ctypes.POINTER(ml_ImageGen)]
_lib.ml_imagegen_destroy.restype = ctypes.c_int32

_lib.ml_imagegen_txt2img.argtypes = [
    ctypes.POINTER(ml_ImageGen),
    ctypes.POINTER(ml_ImageGenTxt2ImgInput),
    ctypes.POINTER(ml_ImageGenOutput),
]
_lib.ml_imagegen_txt2img.restype = ctypes.c_int32

_lib.ml_imagegen_img2img.argtypes = [
    ctypes.POINTER(ml_ImageGen),
    ctypes.POINTER(ml_ImageGenImg2ImgInput),
    ctypes.POINTER(ml_ImageGenOutput),
]
_lib.ml_imagegen_img2img.restype = ctypes.c_int32
