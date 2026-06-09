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

"""On-device geniex_benchmark scorecard run for QDC Android phones.

The host (this pytest process) fetches each model, adb-pushes it, builds the
matrix.tsv, then runs geniex_benchmark on-device. The per-cell JSON is written
straight to the device's QDC_logs/results, which QDC auto-collects — keeping
run_qdc_jobs.py's download_cells path identical to Linux.
"""

import os
import shutil
import subprocess
import tempfile
import urllib.request
import zipfile
from pathlib import Path

from utils import (
    BUNDLE_PATH,
    HOST_IMAGE,
    HOST_ROWS,
    IMAGE_PATH,
    MODELS_PATH,
    RESULTS_PATH,
    push_bundle_if_needed,
    run_adb_command,
)

TSV = "/data/local/tmp/matrix.tsv"


def _flatten_single_dir(d: Path) -> None:
    entries = list(d.iterdir())
    if len(entries) == 1 and entries[0].is_dir():
        inner = entries[0]
        for child in inner.iterdir():
            shutil.move(str(child), str(d / child.name))
        inner.rmdir()


def _fetch_and_push(
    host_tmp: Path, name: str, url: str, kind: str, mmproj_url: str
) -> tuple[str, str] | None:
    """Download a model (+ optional mmproj) on the host, push it to the device.

    Returns (model_path, mmproj_path) device paths; mmproj_path is "" when absent.
    """
    dev_dir = f"{MODELS_PATH}/{name}"
    run_adb_command(f"mkdir -p {dev_dir}")
    try:
        if kind == "bundle":
            local_zip = host_tmp / f"{name}.zip"
            urllib.request.urlretrieve(url, local_zip)
            local_dir = host_tmp / name
            with zipfile.ZipFile(local_zip) as z:
                z.extractall(local_dir)
            _flatten_single_dir(local_dir)
            subprocess.run(
                ["adb", "push", str(local_dir), f"{dev_dir}/bundle"], check=True
            )
            return f"{dev_dir}/bundle", ""
        # The phone has fast direct internet (and curl); the QDC Appium host
        # does not reach HuggingFace reliably, so fetch gguf on-device.
        dev_gguf = f"{dev_dir}/model.gguf"
        run_adb_command(f"curl -L -fS --retry 3 --retry-delay 5 -o {dev_gguf} '{url}'")
        dev_mmproj = ""
        if mmproj_url:
            dev_mmproj = f"{dev_dir}/mmproj.gguf"
            run_adb_command(
                f"curl -L -fS --retry 3 --retry-delay 5 -o {dev_mmproj} '{mmproj_url}'"
            )
        return dev_gguf, dev_mmproj
    except Exception as e:  # noqa: BLE001 — one bad model must not abort the matrix
        print(f"WARNING: {name} fetch/push failed, skipping: {e}")
        return None


def test_scorecard():
    push_bundle_if_needed()
    run_adb_command(f"mkdir -p {MODELS_PATH} {RESULTS_PATH}")

    subprocess.run(["adb", "push", HOST_IMAGE, IMAGE_PATH], check=True)

    rows = [r for r in Path(HOST_ROWS).read_text().splitlines() if r.strip()]
    tsv_lines = []
    with tempfile.TemporaryDirectory() as td:
        host_tmp = Path(td)
        for row in rows:
            name, plugin, devs, url, kind, mmproj_url, vlm, image = row.split("|")
            pushed = _fetch_and_push(host_tmp, name, url, kind, mmproj_url)
            if pushed is None:
                continue
            mpath, mmpath = pushed
            imgpath = IMAGE_PATH if image == "1" else ""
            for d in devs.split(","):
                tsv_lines.append(
                    f"{name}-{plugin}-{d}\t{plugin}\t{d}\t{mpath}"
                    f"\t\t{mmpath}\t{imgpath}\t{vlm}"
                )

    assert tsv_lines, "no models pushed to device"
    run_adb_command(
        "printf '%s\\n' " + " ".join(f"'{ln}'" for ln in tsv_lines) + f" > {TSV}"
    )

    lib = f"{BUNDLE_PATH}/lib"
    env = (
        f"LD_LIBRARY_PATH={lib}:{lib}/llama_cpp:{lib}/qairt "
        f"ADSP_LIBRARY_PATH={lib} "
        f"GENIEX_PLUGIN_PATH={lib}"
    )
    res = run_adb_command(
        f"cd {BUNDLE_PATH} && {env} ./bin/geniex_benchmark "
        f"--matrix-file {TSV} --output-json-dir {RESULTS_PATH} -r 3",
        check=False,
    )
    count = run_adb_command(f"ls {RESULTS_PATH} | wc -l", check=False).stdout.strip()
    assert res.returncode == 0, f"geniex_benchmark exited {res.returncode}"
    assert count and int(count.split()[-1]) > 0, "no cell JSON produced on device"


if __name__ == "__main__":
    import pytest

    raise SystemExit(
        pytest.main(["-s", "--junitxml=results.xml", os.path.realpath(__file__)])
    )
