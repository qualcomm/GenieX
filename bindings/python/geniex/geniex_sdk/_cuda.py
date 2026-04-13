# This module is no longer active. CUDA support has been removed."""
CUDA library preloader for nexa-sdk.

This module preloads NVIDIA CUDA runtime libraries into the global symbol table
so that plugins (ONNX Runtime CUDA, llama.cpp CUDA) can find them at runtime.

This allows users to install CUDA support via pip without requiring system-wide
CUDA installation:
    pip install geniex[cuda]
"""

import ctypes
import logging
import os
import platform

logger = logging.getLogger(__name__)

cuda_found = False


def _find_cuda() -> None:
    """
    Preload CUDA 12 libraries into global symbol table.

    Called automatically at module import time. If CUDA packages are not
    installed, this silently returns without error.
    """
    global cuda_found

    # Only relevant for Linux and Windows
    if platform.system() not in ('Linux', 'Windows'):
        logger.debug('Skipping CUDA preload on non-Linux/Windows platform')
        return

    try:
        # Try to import NVIDIA CUDA packages (optional dependencies)
        from nvidia import cublas, cuda_runtime, cudnn
    except ImportError:
        logger.debug('NVIDIA CUDA packages not found (optional)')
        return

    try:
        if platform.system() == 'Linux':
            cudalib = 'lib/libcudart.so.12'
            cublaslib = 'lib/libcublas.so.12'
            cudnnlib = 'lib/libcudnn.so.9'
        else:  # Windows
            cudalib = r'bin\cudart64_12.dll'
            cublaslib = r'bin\cublas64_12.dll'
            cudnnlib = r'bin\cudnn64_9.dll'

        # Preload with RTLD_GLOBAL flag
        # This makes CUDA symbols available to all subsequently loaded libraries,
        # including ONNX Runtime CUDA and llama.cpp CUDA plugins
        cuda_rt_path = os.path.join(cuda_runtime.__path__[0], cudalib)
        cublas_path = os.path.join(cublas.__path__[0], cublaslib)
        cudnn_path = os.path.join(cudnn.__path__[0], cudnnlib)

        logger.debug(f'Attempting to load CUDA runtime from: {cuda_rt_path}')
        ctypes.CDLL(cuda_rt_path, mode=ctypes.RTLD_GLOBAL)

        logger.debug(f'Attempting to load cuBLAS from: {cublas_path}')
        ctypes.CDLL(cublas_path, mode=ctypes.RTLD_GLOBAL)

        logger.debug(f'Attempting to load cuDNN from: {cudnn_path}')
        ctypes.CDLL(cudnn_path, mode=ctypes.RTLD_GLOBAL)

        cuda_found = True
        logger.info('Successfully preloaded CUDA 12 libraries')

    except OSError as e:
        logger.debug(f'Failed to load CUDA 12 libraries: {e}')
