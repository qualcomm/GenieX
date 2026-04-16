"""
Model management functions for geniex library.

This module provides functions for downloading, listing, and managing models
from HuggingFace Hub and the local store.
"""

import logging
import os
from typing import Callable, List, Optional

from .internal.model_hub import ensure_model_downloaded
from .internal.store import Store
from .internal.types import DownloadProgressInfo, ModelInfo


def download_model(
    repo_id: str,
    quant_spec: Optional[str] = None,
    token: Optional[str] = os.getenv('HF_TOKEN', None),
    progress_callback: Optional[Callable[[DownloadProgressInfo], None]] = None,
    local_dir: Optional[str] = None,
    enable_transfer: bool = True,
    show_progress: bool = True,
    custom_endpoint: Optional[str] = None,
    force_download: bool = False,
    **kwargs,
) -> None:
    """
    Download a model from HuggingFace Hub.

    Args:
        repo_id: HuggingFace repository ID (e.g., "microsoft/phi-3-mini-4k-instruct")
        quant_spec: Optional quant name (e.g., "Q4_K_M") to download specific quantization
        token: Optional HuggingFace token for private repositories, default to environment variable HF_TOKEN
        progress_callback: Optional callback function for download progress.
            The callback receives a DownloadProgressInfo dict with the following structure:
            {
                'status': str,  # 'idle', 'downloading', 'completed', 'error'
                'error_message': str,  # Only present if status is 'error'
                'progress': {
                    'total_downloaded': int,  # Bytes downloaded
                    'total_size': int,        # Total bytes to download
                    'percentage': float,      # Progress percentage (0-100)
                    'files_active': int,      # Number of files currently downloading
                    'files_total': int,       # Total number of files
                    'known_total': bool       # Whether total size is known
                },
                'speed': {
                    'bytes_per_second': float,  # Download speed in bytes/sec
                    'formatted': str            # Human readable speed (e.g., "1.2 MB/s")
                },
                'formatting': {
                    'downloaded': str,  # Human readable downloaded size
                    'total_size': str   # Human readable total size
                },
                'timing': {
                    'elapsed_seconds': float,  # Time since download started
                    'eta_seconds': float,      # Estimated time remaining
                    'start_time': float        # Download start timestamp
                }
            }
        local_dir: Optional directory to download the model to. If not provided, uses default cache directory.
        enable_transfer: If True (default), enable hf_transfer for faster downloads.
        show_progress: If True (default), show download progress bar in terminal.
        custom_endpoint: Optional custom HuggingFace Hub endpoint URL.
        force_download: If True, force re-download even if the model already exists locally.
        **kwargs: Extra fields to store in manifest:
            - avatar_url: Optional avatar URL for the model author

    Example:
        >>> def progress(info):
        ...     pct = info['progress']['percentage']
        ...     downloaded = info['formatting']['downloaded']
        ...     speed = info['speed']['formatted']
        ...     print(f"Progress: {pct:.1f}% | {downloaded} | {speed}")
        >>> geniex.download_model("microsoft/phi-3-mini-4k-instruct", progress_callback=progress)
    """
    logger = logging.getLogger('geniex')

    # Use unified internal function to ensure consistent behavior
    manifest = ensure_model_downloaded(
        repo_id=repo_id,
        quant_spec=quant_spec,
        token=token,
        progress_callback=progress_callback,
        local_dir=local_dir,
        enable_transfer=enable_transfer,
        show_progress=show_progress,
        custom_endpoint=custom_endpoint,
        force_download=force_download,
        **kwargs,
    )

    if manifest is not None:
        logger.info(f'Model {repo_id} is ready (already existed or downloaded successfully)')


def list_models() -> List[ModelInfo]:
    """
    List all models downloaded in the local store.

    Returns:
        List of ModelInfo objects containing information about each model

    Example:
        >>> models = geniex.list_models()
        >>> for model in models:
        ...     print(f"{model.repo_id}: {model.model_type} ({model.size_bytes / 1024**3:.2f} GB)")
    """
    store = Store.get()
    manifests = store.list_models()
    model_infos = []
    for manifest in manifests:
        local_path = str(store.model_dir_path() / manifest.name)
        model_infos.append(ModelInfo.from_manifest(manifest, local_path=local_path))
    return model_infos


def remove_model(repo_id: str) -> Optional[ModelInfo]:
    """
    Remove a model from the local store.

    Args:
        repo_id: HuggingFace repository ID (e.g., "microsoft/phi-3-mini-4k-instruct")

    Returns:
        ModelInfo of the removed model, or None if the model was not found

    Example:
        >>> model = geniex.remove_model("microsoft/phi-3-mini-4k-instruct")
        >>> if model:
        ...     print(f"Removed {model.repo_id} ({model.size_bytes} bytes)")
    """
    store = Store.get()

    # Get the manifest before removing to return ModelInfo
    manifest = store.get_manifest(repo_id)
    if manifest is None:
        return None

    local_path = str(store.model_dir_path() / repo_id)
    model_info = ModelInfo.from_manifest(manifest, local_path=local_path)

    if store.remove_model(repo_id):
        return model_info
    return None
