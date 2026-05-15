# Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
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

"""Local AI Hub pull (extracted directory + local zip) — public API.

Drives ``model_manager.pull(hub='localfs', local_path=...)`` against
fabricated AI Hub layouts (`metadata.json` + `*.bin` shards, or a
matching ``.zip`` on disk) and asserts the resulting on-disk store
matches what a remote AI Hub pull would produce: ``plugin_id == 'qairt'``,
the lex-first ``.bin`` is the entrypoint, modality is driven by
``metadata.json::genie.supports_vision``.

Each test uses a unique ``qualcomm/<repo>`` name and cleans up via
``_mm.remove`` so the user's real cache stays unpolluted.
"""

from __future__ import annotations

import json
import os
import struct
import uuid
import zipfile
from pathlib import Path
from typing import Iterator

import pytest

import geniex
from geniex import model_manager as _mm

# ----- helpers --------------------------------------------------------------


def _write_extracted_aihub(
    dir_path: Path,
    *,
    supports_vision: bool,
    bin_a: bytes = b'shard-a-payload-AAAAAAAA',
    bin_b: bytes = b'shard-b-payload-BBBB',
) -> tuple[bytes, bytes]:
    """Write an `aihub.ExtractFlat`-shaped directory: metadata.json + two
    `.bin` shards + a tokenizer. Returns the bytes for diff assertions."""
    meta = {'genie': {'supports_vision': supports_vision}}
    (dir_path / 'metadata.json').write_text(json.dumps(meta))
    (dir_path / 'shard_a.bin').write_bytes(bin_a)
    (dir_path / 'shard_b.bin').write_bytes(bin_b)
    (dir_path / 'tokenizer.json').write_text('{}')
    return bin_a, bin_b


def _build_aihub_zip(
    zip_path: Path,
    *,
    supports_vision: bool,
    weights: bytes,
) -> None:
    """Write a `.zip` whose `metadata.json` is STORED and `weights.bin`
    is DEFLATE — exercises both `LocalRange` and `LocalDeflate`
    BytesSource variants in the same plan."""
    meta = json.dumps({'genie': {'supports_vision': supports_vision}}).encode()
    with zipfile.ZipFile(zip_path, 'w') as z:
        z.writestr(
            zipfile.ZipInfo('metadata.json'),
            meta,
            compress_type=zipfile.ZIP_STORED,
        )
        z.writestr(
            zipfile.ZipInfo('shard_a.bin'),
            b'hello-shard-a',
            compress_type=zipfile.ZIP_STORED,
        )
        z.writestr(
            zipfile.ZipInfo('weights.bin'),
            weights,
            compress_type=zipfile.ZIP_DEFLATED,
        )


def _unique_name(prefix: str) -> str:
    # `qualcomm/...` is the only prefix the AI Hub canonicaliser recognises;
    # using it keeps the on-disk path under `models/qualcomm/` so cleanup
    # via `_mm.remove` works the same way it does for real AI Hub pulls.
    suffix = f'{os.getpid()}-{uuid.uuid4().hex[:8]}'
    return f'qualcomm/test-local-aihub-{prefix}-{suffix}'


@pytest.fixture
def cleanup_model(geniex_session) -> Iterator[list[str]]:
    """Track names so each test's pulled model is removed even on failure."""
    names: list[str] = []
    try:
        yield names
    finally:
        for n in names:
            try:
                _mm.remove(n)
            except geniex.GeniexError:
                pass


# ----- extracted-directory pull --------------------------------------------


def test_pull_from_extracted_dir_produces_qairt_paths(tmp_path, cleanup_model):
    src = tmp_path / 'extracted'
    src.mkdir()
    bin_a, bin_b = _write_extracted_aihub(src, supports_vision=False)
    name = _unique_name('extracted')

    _mm.pull(name, hub='localfs', local_path=str(src))
    cleanup_model.append(name)

    paths = _mm.get_paths(name)
    assert paths.plugin_id == 'qairt'
    # Lex-first .bin is the entrypoint.
    assert os.path.basename(paths.model_path) == 'shard_a.bin'
    assert os.path.isfile(paths.model_path)

    model_dir = Path(paths.model_dir)
    assert (model_dir / 'shard_a.bin').read_bytes() == bin_a
    assert (model_dir / 'shard_b.bin').read_bytes() == bin_b
    assert (model_dir / 'tokenizer.json').read_text() == '{}'
    assert (model_dir / 'metadata.json').exists()
    assert (model_dir / 'geniex.json').exists()

    assert _mm.get_type(name) == 'llm'
    assert name in _mm.list_models()


def test_pull_from_extracted_dir_supports_vision_yields_vlm(tmp_path, cleanup_model):
    src = tmp_path / 'extracted-vlm'
    src.mkdir()
    _write_extracted_aihub(src, supports_vision=True)
    name = _unique_name('extracted-vlm')

    _mm.pull(name, hub='localfs', local_path=str(src))
    cleanup_model.append(name)

    assert _mm.get_type(name) == 'vlm'


# ----- local zip pull -------------------------------------------------------


def test_pull_from_local_zip_handles_stored_and_deflate(tmp_path, cleanup_model):
    # 16 KiB of structured-but-compressible bytes so DEFLATE actually
    # shrinks the entry (zip writers may keep tiny inputs as STORED).
    weights = struct.pack('<H', 0xAA55) * 8192
    zip_path = tmp_path / 'model.zip'
    _build_aihub_zip(zip_path, supports_vision=True, weights=weights)
    name = _unique_name('zip-vlm')

    _mm.pull(name, hub='localfs', local_path=str(zip_path))
    cleanup_model.append(name)

    paths = _mm.get_paths(name)
    assert paths.plugin_id == 'qairt'
    # Lex-first .bin between {shard_a.bin, weights.bin} is shard_a.bin.
    assert os.path.basename(paths.model_path) == 'shard_a.bin'

    model_dir = Path(paths.model_dir)
    assert (model_dir / 'shard_a.bin').read_bytes() == b'hello-shard-a'
    assert (model_dir / 'weights.bin').read_bytes() == weights
    assert (model_dir / 'metadata.json').exists()

    assert _mm.get_type(name) == 'vlm'


# ----- error paths ----------------------------------------------------------


def test_pull_rejects_unknown_local_layout(tmp_path, geniex_session):
    src = tmp_path / 'mystery'
    src.mkdir()
    (src / 'README.md').write_text('hi')
    name = _unique_name('unknown')

    with pytest.raises(geniex.GeniexError):
        _mm.pull(name, hub='localfs', local_path=str(src))


def test_pull_rejects_safetensors_only_dir_with_clear_error(tmp_path, geniex_session):
    src = tmp_path / 'hf-safetensors'
    src.mkdir()
    (src / 'config.json').write_text('{"architectures":["LlamaForCausalLM"]}')
    (src / 'model.safetensors').write_bytes(b'\x00\x00')
    name = _unique_name('safetensors')

    with pytest.raises(geniex.GeniexError):
        _mm.pull(name, hub='localfs', local_path=str(src))


def test_pull_localfs_requires_local_path(geniex_session):
    name = _unique_name('no-path')
    with pytest.raises(geniex.GeniexError):
        _mm.pull(name, hub='localfs')
