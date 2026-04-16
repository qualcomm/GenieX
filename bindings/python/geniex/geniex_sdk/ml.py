"""
ML core functions and structures for nexa-sdk.
"""

import ctypes

from ._lib import _lib
from .error import ml_log_callback

# Plugin function types
ml_plugin_id_func = ctypes.CFUNCTYPE(ctypes.c_char_p)
ml_create_plugin_func = ctypes.CFUNCTYPE(ctypes.c_void_p)


# ============================================================================
# C Structure Definitions
# ============================================================================


class ml_GetPluginListOutput(ctypes.Structure):
    """Output structure containing the list of available plugins."""

    _fields_ = [
        ('plugin_ids', ctypes.POINTER(ctypes.c_char_p)),
        ('plugin_count', ctypes.c_int32),
    ]


class ml_GetDeviceListInput(ctypes.Structure):
    """Input structure for querying available devices for a plugin."""

    _fields_ = [
        ('plugin_id', ctypes.c_char_p),
    ]


class ml_GetDeviceListOutput(ctypes.Structure):
    """Output structure containing the list of available devices."""

    _fields_ = [
        ('device_ids', ctypes.POINTER(ctypes.c_char_p)),
        ('device_names', ctypes.POINTER(ctypes.c_char_p)),
        ('device_count', ctypes.c_int32),
    ]


# ============================================================================
# Function Signatures
# ============================================================================

_lib.ml_init.argtypes = []
_lib.ml_init.restype = ctypes.c_int32

_lib.ml_deinit.argtypes = []
_lib.ml_deinit.restype = ctypes.c_int32

_lib.ml_set_log.argtypes = [ml_log_callback]
_lib.ml_set_log.restype = ctypes.c_int32

_lib.ml_free.argtypes = [ctypes.c_void_p]
_lib.ml_free.restype = None

_lib.ml_version.argtypes = []
_lib.ml_version.restype = ctypes.c_char_p

_lib.ml_register_plugin.argtypes = [ml_plugin_id_func, ml_create_plugin_func]
_lib.ml_register_plugin.restype = ctypes.c_int32

_lib.ml_get_plugin_list.argtypes = [ctypes.POINTER(ml_GetPluginListOutput)]
_lib.ml_get_plugin_list.restype = ctypes.c_int32

_lib.ml_get_device_list.argtypes = [
    ctypes.POINTER(ml_GetDeviceListInput),
    ctypes.POINTER(ml_GetDeviceListOutput),
]
_lib.ml_get_device_list.restype = ctypes.c_int32
