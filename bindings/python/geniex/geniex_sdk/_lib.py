"""
Library loader for nexa-sdk C library.
"""

import ctypes
import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def _find_library() -> Path:
    lib_name = {
        'win32': 'geniex_bridge.dll',
        'linux': 'libgeniex_bridge.so',
    }.get(sys.platform, None)

    if lib_name is None:
        raise RuntimeError(f'Unsupported platform: {sys.platform}')

    lib_paths = [
        Path(__file__).parent.parent.parent.parent.parent / 'build' / 'out' / lib_name,
        Path(os.getenv('GENIEX_PLUGIN_PATH', Path(__file__).parent / 'lib')) / lib_name,
    ]

    for lib_path in lib_paths:
        logger.debug(f'Checking library: {lib_path}')
        if lib_path.exists():
            return lib_path

    raise RuntimeError('Could not find nexa bridge library')


# Load the library
_lib_path = _find_library()
logger.debug(f'Loading library from: {_lib_path}')
_lib = ctypes.CDLL(str(_lib_path))
