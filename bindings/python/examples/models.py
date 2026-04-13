# Copyright 2024-2026 Qualcomm
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Example script demonstrating how to use the model management API.

Usage:
    python models.py [--model MODEL_ID] [--quant QUANT_SPEC] [--token HF_TOKEN] [--no-progress]
"""

import argparse
import logging
import time
from typing import Dict

from tqdm import tqdm

from geniex import (
    DownloadProgressInfo,
    download_model,
    get_plugin_list,
    list_models,
    geniex_version,
    remove_model,
    setup_logging,
    version,
)

_progress_bars: Dict[str, tqdm] = {}


def progress_callback(info: DownloadProgressInfo) -> None:
    print(info)


def main():
    parser = argparse.ArgumentParser(description='GenieX-Bridge Model Management Examples')
    parser.add_argument('-m', '--model', default='Geniex/yolov12-npu', help='HuggingFace repository ID')
    parser.add_argument('--quant', default=None, help='Quantization specification (e.g., Q4_K_M)')
    parser.add_argument('--token', default=None, help='HuggingFace token for private repos')
    parser.add_argument('--no-progress', action='store_true', help='Disable progress bar')
    args = parser.parse_args()

    setup_logging(logging.NOTSET)
    print('GenieX-Bridge Model Management Examples')
    print('=' * 50)
    print(f'Python Version: {version()}')
    print(f'SDK Version: {geniex_version()}\n')
    print(f'Available plugins: {", ".join(get_plugin_list())}\n')

    repo_id = args.model
    if any(m.repo_id == repo_id for m in list_models()):
        print(f'Removing existing model {repo_id}...')
        remove_model(repo_id)
        print()

    print(f'Downloading model: {repo_id}')
    if args.quant:
        print(f'Quantization: {args.quant}')
    print()

    try:
        download_model(
            repo_id=repo_id,
            quant_spec=args.quant,
            token=args.token,
            progress_callback=None if args.no_progress else progress_callback,
        )
        time.sleep(0.5)
        print(f'\n✓ Successfully downloaded {repo_id}')
    finally:
        for pbar in _progress_bars.values():
            if not pbar.disable:
                pbar.close()
        _progress_bars.clear()

    models = list_models()
    print(f'\nFound {len(models)} model(s) in local store:\n')
    for m in models:
        size_gb = m.size / (1024**3) if m.size > 0 else 0
        print(f'  Repository ID: {m.repo_id}')
        print(f'  Model Name: {m.model_name}')
        print(f'  Model Type: {m.model_type}')
        print(f'  Size: {size_gb:.2f} GB')
        print(f'  Plugin ID: {m.plugin_id or "default"}')
        print(f'  Device ID: {m.device_id or "default"}\n')

    print(f'Removing model: {repo_id}')
    success = remove_model(repo_id)
    print(f'{"✓" if success else "✗"} {"Successfully removed" if success else "Failed to remove"} {repo_id}')

    print('\n' + '=' * 50)
    print('Examples completed!')


if __name__ == '__main__':
    main()
