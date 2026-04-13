"""
Core functions and logging configuration for geniex library.
Matches C logging format: [LEVEL] [filename:line:func] message
"""

import ctypes
import logging
import os
import sys
from typing import List, Optional, Tuple

from .geniex_sdk._lib import _lib
from .geniex_sdk.error import (
    ML_LOG_LEVEL_DEBUG,
    ML_LOG_LEVEL_ERROR,
    ML_LOG_LEVEL_INFO,
    ML_LOG_LEVEL_TRACE,
    ML_LOG_LEVEL_WARN,
    check_error,
    ml_log_callback,
)
from .geniex_sdk.ml import ml_GetDeviceListInput, ml_GetDeviceListOutput, ml_GetPluginListOutput

_C_LOG_LEVEL_TO_PYTHON = {
    ML_LOG_LEVEL_TRACE: logging.DEBUG,
    ML_LOG_LEVEL_DEBUG: logging.DEBUG,
    ML_LOG_LEVEL_INFO: logging.DEBUG,
    ML_LOG_LEVEL_WARN: logging.WARNING,
    ML_LOG_LEVEL_ERROR: logging.ERROR,
}

_LEVEL_NAMES = {
    logging.DEBUG: 'DEBUG',
    logging.INFO: 'INFO',
    logging.WARNING: 'WARN',
    logging.ERROR: 'ERROR',
    logging.CRITICAL: 'CRITICAL',
}

_COLORS = {
    'reset': '\033[0m',
    'cyan': '\033[36m',
    'green': '\033[32m',
    'yellow': '\033[33m',
    'red': '\033[31m',
}

_LEVEL_COLORS = {
    logging.DEBUG: _COLORS['cyan'],
    logging.INFO: _COLORS['green'],
    logging.WARNING: _COLORS['yellow'],
    logging.ERROR: _COLORS['red'],
    logging.CRITICAL: _COLORS['red'],
}

_log_callback_ref = None


def get_plugin_list() -> List[str]:
    """Query the list of available plugins."""
    output = ml_GetPluginListOutput()
    error_code = _lib.ml_get_plugin_list(ctypes.pointer(output))
    check_error(error_code)
    plugin_ids = []
    if output.plugin_ids and output.plugin_count > 0:
        for i in range(output.plugin_count):
            plugin_id_ptr = output.plugin_ids[i]
            if plugin_id_ptr:
                plugin_id = plugin_id_ptr.decode('utf-8')
                plugin_ids.append(plugin_id)
        _lib.ml_free(output.plugin_ids)
    return plugin_ids


def get_device_list(plugin_id: str) -> Tuple[List[str], List[str]]:
    """Query the list of available devices for a given plugin."""
    input_struct = ml_GetDeviceListInput(plugin_id=plugin_id.encode('utf-8'))
    output = ml_GetDeviceListOutput()
    error_code = _lib.ml_get_device_list(ctypes.pointer(input_struct), ctypes.pointer(output))
    check_error(error_code)
    device_ids = []
    device_names = []
    if output.device_count > 0:
        if output.device_ids:
            for i in range(output.device_count):
                device_id_ptr = output.device_ids[i]
                if device_id_ptr:
                    device_id = device_id_ptr.decode('utf-8')
                    device_ids.append(device_id)
        if output.device_names:
            for i in range(output.device_count):
                device_name_ptr = output.device_names[i]
                if device_name_ptr:
                    device_name = device_name_ptr.decode('utf-8')
                    device_names.append(device_name)
        if output.device_ids:
            _lib.ml_free(output.device_ids)
        if output.device_names:
            _lib.ml_free(output.device_names)
    return device_ids, device_names


class Geniex(logging.Formatter):
    """Formatter matching C style: [LEVEL] [filename:line:func] message"""

    def __init__(self, use_colors: bool = True):
        super().__init__()
        self.use_colors = use_colors and hasattr(sys.stdout, 'isatty') and sys.stdout.isatty()

    def format(self, record: logging.LogRecord) -> str:
        level_name = _LEVEL_NAMES.get(record.levelno, record.levelname)
        filename = os.path.basename(record.pathname) if hasattr(record, 'pathname') else record.filename
        line = record.lineno
        func = record.funcName
        msg = record.getMessage()

        formatted = f'[{level_name:5}] [{filename}:{line}:{func}] {msg}'

        if self.use_colors:
            level_color = _LEVEL_COLORS.get(record.levelno, '')
            reset = _COLORS['reset']
            return f'{level_color}[{level_name:5}] [{filename}:{line}{level_color}:{func}] {msg}{reset}'

        return formatted


def setup_logging(level: int = logging.INFO, fmt: Optional[logging.Formatter] = None) -> None:
    formatter = fmt if fmt is not None else Geniex()
    handler = logging.StreamHandler()
    handler.setLevel(level)
    handler.setFormatter(formatter)

    logger = logging.getLogger('geniex')
    logger.setLevel(level)
    logger.addHandler(handler)


def _init_logging() -> None:
    global _log_callback_ref

    sdk_logger = logging.getLogger('geniex.geniex_sdk')

    def _log_callback(c_level: int, message: ctypes.c_char_p) -> None:
        if not message:
            return
        try:
            msg = message.decode('utf-8', errors='replace')
            sdk_logger.log(_C_LOG_LEVEL_TO_PYTHON.get(c_level, logging.DEBUG), msg)
        except Exception:
            sdk_logger.error(f'Failed to decode log message: {message}')
            pass

    _log_callback_ref = ml_log_callback(_log_callback)
    check_error(_lib.ml_set_log(_log_callback_ref))
