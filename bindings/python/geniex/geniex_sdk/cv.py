"""
CV functions and structures for nexa-sdk.
"""

import ctypes

from ._lib import _lib
from .error import ml_log_callback
from .llm import ml_ModelConfig

# ============================================================================
# C Structure Definitions
# ============================================================================


class ml_BoundingBox(ctypes.Structure):
    """Generic bounding box structure."""

    _fields_ = [
        ('x', ctypes.c_float),
        ('y', ctypes.c_float),
        ('width', ctypes.c_float),
        ('height', ctypes.c_float),
    ]


class ml_CVResult(ctypes.Structure):
    """Generic detection/classification result."""

    _fields_ = [
        ('image_paths', ctypes.POINTER(ctypes.c_char_p)),
        ('image_count', ctypes.c_int32),
        ('class_id', ctypes.c_int32),
        ('confidence', ctypes.c_float),
        ('bbox', ml_BoundingBox),
        ('text', ctypes.c_char_p),
        ('embedding', ctypes.POINTER(ctypes.c_float)),
        ('embedding_dim', ctypes.c_int32),
        ('mask', ctypes.POINTER(ctypes.c_float)),
        ('mask_h', ctypes.c_int32),
        ('mask_w', ctypes.c_int32),
    ]


class ml_CVCapabilities(ctypes.c_int32):
    """CV capabilities enum."""

    ML_CV_OCR = 0
    ML_CV_CLASSIFICATION = 1
    ML_CV_SEGMENTATION = 2
    ML_CV_CUSTOM = 3


class ml_CVModelConfig(ctypes.Structure):
    """CV model preprocessing configuration."""

    _fields_ = [
        ('capabilities', ml_CVCapabilities),
        ('det_model_path', ctypes.c_char_p),
        ('rec_model_path', ctypes.c_char_p),
        ('char_dict_path', ctypes.c_char_p),
        ('qnn_model_folder_path', ctypes.c_char_p),
        ('qnn_lib_folder_path', ctypes.c_char_p),
    ]


class ml_CV(ctypes.Structure):
    """Opaque CV model handle."""

    pass


class ml_CVCreateInput(ctypes.Structure):
    """Input structure for creating a CV model."""

    _fields_ = [
        ('model_name', ctypes.c_char_p),
        ('config', ml_CVModelConfig),
        ('plugin_id', ctypes.c_char_p),
        ('device_id', ctypes.c_char_p),
        ('license_id', ctypes.c_char_p),
        ('license_key', ctypes.c_char_p),
    ]


class ml_CVInferInput(ctypes.Structure):
    """Input structure for CV inference."""

    _fields_ = [
        ('input_image_path', ctypes.c_char_p),
    ]


class ml_CVInferOutput(ctypes.Structure):
    _fields_ = [
        ('results', ctypes.POINTER(ml_CVResult)),
        ('result_count', ctypes.c_int32),
    ]


# ============================================================================
# Function Signatures
# ============================================================================

_lib.ml_cv_create.argtypes = [ctypes.POINTER(ml_CVCreateInput), ctypes.POINTER(ctypes.POINTER(ml_CV))]
_lib.ml_cv_create.restype = ctypes.c_int32

_lib.ml_cv_destroy.argtypes = [ctypes.POINTER(ml_CV)]
_lib.ml_cv_destroy.restype = ctypes.c_int32

_lib.ml_cv_infer.argtypes = [
    ctypes.POINTER(ml_CV),
    ctypes.POINTER(ml_CVInferInput),
    ctypes.POINTER(ml_CVInferOutput),
]
_lib.ml_cv_infer.restype = ctypes.c_int32
