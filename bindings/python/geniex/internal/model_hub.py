import json
import logging
import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from huggingface_hub import HfApi

from .progress_tracker import DownloadProgressTracker
from .types import DownloadProgressInfo, ModelFileInfo, ModelManifest

logger = logging.getLogger(__name__)

# Quant regex pattern matching Go implementation
# Matches: FP32, FP16, F64, F32, F16, I64, I32, I16, I8,
#          Q8_0, Q8_1, Q8_K, Q6_K, Q5_0, Q5_1, Q5_K, Q4_0, Q4_1, Q4_K, Q3_K, Q2_K,
#          IQ4_NL, IQ4_XS, IQ3_S, IQ3_XXS, IQ2_XXS, IQ2_S, IQ2_XS, IQ1_S, IQ1_M,
#          BF16, 1bit, 2bit, 3bit, 4bit, 16bit, etc.
QUANT_REGEX = re.compile(
    r'('
    r'[fF][pP][0-9]+|'  # FP32, FP16, FP64
    r'[fF][0-9]+|'  # F64, F32, F16
    r'[iI][0-9]+|'  # I64, I32, I16, I8
    r'[qQ][0-9]+(_[A-Za-z0-9]+)*|'  # Q8_0, Q8_1, Q8_K, Q6_K, Q5_0, Q5_1, Q5_K, Q4_0, Q4_1, Q4_K, Q3_K, Q2_K
    r'[iI][qQ][0-9]+(_[A-Za-z0-9]+)*|'  # IQ4_NL, IQ4_XS, IQ3_S, IQ3_XXS, IQ2_XXS, IQ2_S, IQ2_XS, IQ1_S, IQ1_M
    r'[bB][fF][0-9]+|'  # BF16
    r'[0-9]+[bB][iI][tT]'  # 1bit, 2bit, 3bit, 4bit, 16bit, 1BIT, 16Bit, etc.
    r')'
)

# Regex to match part files like "-00003-of-00003.gguf"
PART_REGEX = re.compile(r'-\d+-of-\d+\.gguf$')


def get_quant(name: str) -> str:
    match = QUANT_REGEX.search(name)
    if match:
        return match.group(0).upper()
    return 'N/A'


def sum_size(files: List[ModelFileInfo]) -> int:
    return sum(f.size for f in files)


def _update_manifest_file_info(manifest: ModelManifest, filename: str, size: int) -> None:
    # Check all possible locations
    for mf_info in manifest.model_file.values():
        if mf_info.name == filename:
            mf_info.size = size
            mf_info.downloaded = True
            return

    if manifest.mmproj_file.name == filename:
        manifest.mmproj_file.size = size
        manifest.mmproj_file.downloaded = True
        return

    if manifest.tokenizer_file.name == filename:
        manifest.tokenizer_file.size = size
        manifest.tokenizer_file.downloaded = True
        return

    for extra_file in manifest.extra_files:
        if extra_file.name == filename:
            extra_file.size = size
            extra_file.downloaded = True
            return


def choose_files(
    repo_id: str,
    specified_quant: Optional[str],
    files: List[ModelFileInfo],
    existing_manifest: Optional[ModelManifest] = None,
    model_type: Optional[str] = None,
) -> ModelManifest:
    if not files:
        raise ValueError('Repository is empty')

    # Extract model name from repo_id
    model_name = repo_id.split('/')[-1] if '/' in repo_id else repo_id

    # Determine model_type: priority: existing_manifest > passed parameter > default 'llm'
    final_model_type = 'llm'  # Default
    if existing_manifest and existing_manifest.model_type:
        final_model_type = existing_manifest.model_type
    elif model_type:
        final_model_type = model_type

    manifest = ModelManifest(
        name=repo_id,
        model_name=model_name,
        model_type=final_model_type,
    )

    # Copy metadata from existing manifest if available
    if existing_manifest:
        manifest.model_name = existing_manifest.model_name
        manifest.plugin_id = existing_manifest.plugin_id
        manifest.device_id = existing_manifest.device_id
        manifest.model_type = existing_manifest.model_type  # Keep from manifest if exists
        manifest.min_sdk_version = existing_manifest.min_sdk_version

    # Classify files
    mmprojs: List[ModelFileInfo] = []
    tokenizers: List[ModelFileInfo] = []
    onnx_files: List[ModelFileInfo] = []
    nexa_files: List[ModelFileInfo] = []
    npy_files: List[ModelFileInfo] = []
    ggufs: Dict[str, List[ModelFileInfo]] = {}  # key is gguf name without part

    for file_info in files:
        name_lower = file_info.name.lower()
        if name_lower.endswith('.gguf'):
            if name_lower.startswith('mmproj'):
                mmprojs.append(file_info)
            else:
                # Remove part suffix (e.g., "-00003-of-00003.gguf")
                base_name = PART_REGEX.sub('', file_info.name)
                if base_name not in ggufs:
                    ggufs[base_name] = []
                ggufs[base_name].append(file_info)
        elif name_lower.endswith('tokenizer.json'):
            tokenizers.append(file_info)
        elif name_lower.endswith('.onnx'):
            onnx_files.append(file_info)
        elif name_lower.endswith('.nexa'):
            nexa_files.append(file_info)
        elif name_lower.endswith('.npy'):
            npy_files.append(file_info)

    # Choose model files
    if ggufs:
        # Handle GGUF files
        if len(ggufs) == 1:
            # Single quant
            base_name, gguf_list = next(iter(ggufs.items()))
            quant_name = get_quant(base_name) or 'default'
            gguf_list_sorted = sorted(gguf_list, key=lambda f: f.name)

            manifest.model_file[quant_name] = ModelFileInfo(
                name=gguf_list_sorted[0].name,
                downloaded=True,
                size=sum_size(gguf_list),
            )
            # Add other fragments as extra files
            manifest.extra_files.extend(
                ModelFileInfo(name=f.name, downloaded=True, size=f.size) for f in gguf_list_sorted[1:]
            )

            if specified_quant and specified_quant not in manifest.model_file:
                raise ValueError(f'Specified quant {specified_quant} not found')
        else:
            # Multiple quants - choose one
            gguf_names = list(ggufs.keys())

            # Sort by size (largest first)
            gguf_names.sort(key=lambda k: sum_size(ggufs[k]), reverse=True)

            # Find default file (prefer Q8_0, Q4_K_M, Q4_0)
            default_file = None
            preferred_quants = ['Q8_0', 'Q4_K_M', 'Q4_0']
            for gguf_name in gguf_names:
                quant = get_quant(gguf_name)
                if quant in preferred_quants:
                    default_quant = get_quant(default_file) if default_file else None
                    if default_file is None or preferred_quants.index(quant) < preferred_quants.index(default_quant):
                        default_file = gguf_name

            if default_file is None:
                default_file = gguf_names[0]

            # If specified quant, find matching file
            selected_file = default_file
            if specified_quant:
                selected_file = None
                for gguf_name in gguf_names:
                    if get_quant(gguf_name).upper() == specified_quant.upper():
                        selected_file = gguf_name
                        break
                if selected_file is None:
                    available_quants = [get_quant(name) for name in gguf_names]
                    raise ValueError(
                        f'Specified quant {specified_quant} not found. Available quants: {available_quants}'
                    )

            # Create manifest entries for all quants
            for gguf_name, gguf_list in ggufs.items():
                quant_name = get_quant(gguf_name) or 'default'
                gguf_list_sorted = sorted(gguf_list, key=lambda f: f.name)
                downloaded = gguf_name == selected_file

                manifest.model_file[quant_name] = ModelFileInfo(
                    name=gguf_list_sorted[0].name,
                    downloaded=downloaded,
                    size=sum_size(gguf_list),
                )
                # Add other fragments as extra files
                manifest.extra_files.extend(
                    ModelFileInfo(name=f.name, downloaded=downloaded, size=f.size) for f in gguf_list_sorted[1:]
                )

        # Handle mmproj
        if not mmprojs:
            # Fallback to onnx or nexa file as mmproj
            if len(onnx_files) == 1:
                manifest.mmproj_file = ModelFileInfo(name=onnx_files[0].name, downloaded=True, size=onnx_files[0].size)
            elif len(nexa_files) == 1:
                manifest.mmproj_file = ModelFileInfo(name=nexa_files[0].name, downloaded=True, size=nexa_files[0].size)
        elif len(mmprojs) == 1:
            manifest.mmproj_file = ModelFileInfo(name=mmprojs[0].name, downloaded=True, size=mmprojs[0].size)
        else:
            # Choose biggest
            biggest = max(mmprojs, key=lambda f: f.size)
            manifest.mmproj_file = ModelFileInfo(name=biggest.name, downloaded=True, size=biggest.size)

        # Handle tokenizer
        if len(tokenizers) == 1:
            manifest.tokenizer_file = ModelFileInfo(
                name=tokenizers[0].name,
                downloaded=True,
                size=tokenizers[0].size,
            )
        elif len(tokenizers) > 1:
            raise ValueError(
                f'Multiple tokenizer files found: {[t.name for t in tokenizers]}. Expected exactly one tokenizer file'
            )

        # Add .nexa files as extra files (except if used as mmproj)
        manifest.extra_files.extend(
            ModelFileInfo(name=f.name, downloaded=True, size=f.size)
            for f in nexa_files
            if manifest.mmproj_file.name != f.name
        )

        # Add .npy files as extra files
        manifest.extra_files.extend(ModelFileInfo(name=f.name, downloaded=True, size=f.size) for f in npy_files)
    else:
        # Non-GGUF models
        if specified_quant:
            raise ValueError(f'Specified quant {specified_quant} only supported in GGUF models')

        # Extract quant from repo name
        quant = get_quant(repo_id)

        # Find main model file (prefer non-nested files)
        supported_extensions = ['.safetensors', '.npz', '.nexa', '.bin']
        main_file = None
        for file_info in files:
            name_lower = file_info.name.lower()
            if any(name_lower.endswith(ext) for ext in supported_extensions):
                if '/' not in file_info.name:
                    main_file = file_info
                    break

        # Fallback to any supported file, then first file
        if main_file is None:
            for file_info in files:
                name_lower = file_info.name.lower()
                if any(name_lower.endswith(ext) for ext in supported_extensions):
                    main_file = file_info
                    break
            if main_file is None:
                main_file = files[0]

        manifest.model_file[quant] = ModelFileInfo(
            name=main_file.name,
            downloaded=True,
            size=main_file.size,
        )

        # Add other files as extra files
        manifest.extra_files.extend(
            ModelFileInfo(name=f.name, downloaded=True, size=f.size) for f in files if f.name != main_file.name
        )

    # Set default values
    if not manifest.model_name:
        manifest.model_name = repo_id.split('/')[-1] if '/' in repo_id else repo_id

    if not manifest.plugin_id:
        manifest.plugin_id = 'cpu_gpu'

    return manifest


def is_local_path(path: str) -> bool:
    if os.path.isabs(path) or os.path.exists(path) or '\\' in path:
        return True
    if '/' in path:
        if path.count('/') == 1:
            return path.startswith('./') or path.startswith('../') or os.path.exists(path)
        return True
    return os.path.exists(path)


def get_model_info(
    repo_id: str,
    token: Optional[str] = None,
    custom_endpoint: Optional[str] = None,
) -> Tuple[List[ModelFileInfo], Optional[ModelManifest], Optional[str]]:
    api = HfApi(token=token, endpoint=custom_endpoint) if custom_endpoint else HfApi(token=token)

    try:
        files = api.list_repo_files(repo_id=repo_id, repo_type='model')
    except Exception as e:
        logger.error(f'Failed to list files for {repo_id}: {e}')
        raise

    file_size_map = {}
    pipeline_tag = None
    try:
        repo_info = api.repo_info(repo_id=repo_id, repo_type='model', files_metadata=True)
        if repo_info:
            # Get pipeline_tag from model info
            if hasattr(repo_info, 'pipeline_tag') and repo_info.pipeline_tag:
                pipeline_tag = repo_info.pipeline_tag
            # Get file sizes
            if hasattr(repo_info, 'siblings') and repo_info.siblings:
                for sibling in repo_info.siblings:
                    if hasattr(sibling, 'rfilename') and hasattr(sibling, 'size'):
                        file_size_map[sibling.rfilename] = sibling.size
    except Exception as e:
        logger.warning(f'Failed to get repo info for {repo_id}: {e}')

    file_infos = [ModelFileInfo(name=f, downloaded=False, size=file_size_map.get(f, 0)) for f in files]
    has_manifest = 'nexa.manifest' in files

    manifest = None
    if has_manifest:
        try:
            manifest_path = api.hf_hub_download(repo_id=repo_id, filename='nexa.manifest', repo_type='model')
            with open(manifest_path, 'r', encoding='utf-8') as f:
                manifest = ModelManifest.from_dict(json.load(f))
        except Exception as e:
            logger.warning(f'Failed to download manifest for {repo_id}: {e}')

    return file_infos, manifest, pipeline_tag


def download_model(
    repo_id: str,
    output_dir: Path,
    files: List[ModelFileInfo],
    token: Optional[str] = None,
    quant_spec: Optional[str] = None,
    existing_manifest: Optional[ModelManifest] = None,
    progress_callback: Optional[Callable[[DownloadProgressInfo], None]] = None,
    enable_transfer: bool = True,
    show_progress: bool = True,
    custom_endpoint: Optional[str] = None,
    force_download: bool = False,
    model_type: Optional[str] = None,
    **kwargs,
) -> ModelManifest:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = choose_files(repo_id, quant_spec, files, existing_manifest, model_type=model_type)

    files_to_download = [f for f in manifest.model_file.values() if f.downloaded and f.name]
    if manifest.mmproj_file.downloaded and manifest.mmproj_file.name:
        files_to_download.append(manifest.mmproj_file)
    if manifest.tokenizer_file.downloaded and manifest.tokenizer_file.name:
        files_to_download.append(manifest.tokenizer_file)
    files_to_download.extend(f for f in manifest.extra_files if f.downloaded and f.name)

    logger.debug(f'Downloading {len(files_to_download)} files from {repo_id}...')

    os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1' if enable_transfer else '0'
    hf_token = token or os.getenv('HF_TOKEN')
    api = HfApi(token=hf_token, endpoint=custom_endpoint) if custom_endpoint else HfApi(token=hf_token)
    total_size = sum(f.size for f in files_to_download)
    file_count = len(files_to_download)

    tracker: Optional[DownloadProgressTracker] = None
    if progress_callback:
        tracker = DownloadProgressTracker(
            progress_callback=progress_callback,
            show_progress=show_progress,
        )
        tracker.set_repo_info(total_size, file_count)
        tracker.start_tracking()

    try:
        for file_info in files_to_download:
            try:
                expected_path = output_dir / file_info.name

                # Skip existing files unless force_download is True
                if not force_download and expected_path.exists() and file_info.size > 0:
                    actual_size = expected_path.stat().st_size
                    if actual_size == file_info.size:
                        logger.debug(f'File already exists: {file_info.name} (size: {actual_size})')
                        _update_manifest_file_info(manifest, file_info.name, actual_size)
                        continue

                path = api.hf_hub_download(
                    repo_id=repo_id,
                    filename=file_info.name,
                    local_dir=str(output_dir),
                    force_download=force_download,
                )
                if path and os.path.exists(path):
                    expected_path.parent.mkdir(parents=True, exist_ok=True)
                    if os.path.isdir(path) and not expected_path.exists():
                        shutil.copytree(path, expected_path, dirs_exist_ok=True)
                    elif path != str(expected_path):
                        shutil.copy2(path, expected_path)

                actual_size = expected_path.stat().st_size if expected_path.exists() else 0
                _update_manifest_file_info(manifest, file_info.name, actual_size)
                logger.debug(f'Downloaded: {file_info.name} to {expected_path} (size: {actual_size})')
            except Exception as e:
                logger.warning(f'Failed to download {file_info.name}: {e}')
                if tracker:
                    tracker.set_error(f'Failed to download {file_info.name}: {e}')
    finally:
        if tracker:
            tracker.stop_tracking()

    manifest.pipeline_tag = kwargs.get('pipeline_tag')
    manifest.avatar_url = kwargs.get('avatar_url')
    manifest.download_time = datetime.now(timezone.utc).isoformat()
    return manifest


def ensure_model_downloaded(
    repo_id: str,
    quant_spec: Optional[str] = None,
    token: Optional[str] = None,
    progress_callback: Optional[Callable[[DownloadProgressInfo], None]] = None,
    local_dir: Optional[str] = None,
    enable_transfer: bool = True,
    show_progress: bool = True,
    custom_endpoint: Optional[str] = None,
    force_download: bool = False,
    model_type: Optional[str] = None,
    **kwargs,
) -> 'ModelManifest':
    from .store import Store

    store = Store.get()
    model_dir = Path(local_dir) / repo_id if local_dir else store.model_dir_path() / repo_id
    manifest = store.get_manifest(repo_id) if not force_download and store.get_manifest(repo_id) else None
    if manifest:
        logger.debug(f'Model {repo_id} already exists locally')
        return manifest

    logger.debug(f'Model {repo_id} not found locally, downloading from Huggingface...')
    if not store.lock_model(repo_id):
        raise RuntimeError(f'Failed to acquire lock for model {repo_id}')

    try:
        files, existing_manifest, pipeline_tag = get_model_info(repo_id, token=token, custom_endpoint=custom_endpoint)
        manifest = download_model(
            repo_id=repo_id,
            output_dir=model_dir,
            files=files,
            token=token,
            quant_spec=quant_spec,
            existing_manifest=existing_manifest,
            progress_callback=progress_callback,
            pipeline_tag=pipeline_tag,
            model_type=model_type,
            enable_transfer=enable_transfer,
            show_progress=show_progress,
            custom_endpoint=custom_endpoint,
            force_download=force_download,
            **kwargs,
        )
        store.save_manifest(manifest)
    finally:
        store.unlock_model(repo_id)

    return manifest
