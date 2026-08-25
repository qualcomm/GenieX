# Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""On-device geniex-bench run for QDC Android phones.

The host (this pytest process) builds the matrix.tsv with model-manager ids
in column 4 and runs geniex-bench on-device; the benchmark resolves each id
via the model-manager C API and downloads to GENIEX_DATADIR on first use,
replacing the host-side urllib + on-device curl that earlier revisions used.
The per-cell JSON is written straight to the device's QDC_logs/results,
which QDC auto-collects — keeping run_qdc_jobs.py's download_cells path
identical to Linux.
"""

import os
import subprocess
from pathlib import Path

from utils import (
    BUNDLE_PATH,
    HOST_CHIPSET,
    HOST_IMAGE,
    HOST_PROMPTS,
    HOST_ROWS,
    IMAGE_PATH,
    MM_CACHE_PATH,
    PROMPTS_PATH,
    RESULTS_PATH,
    push_bundle_if_needed,
    run_adb_command,
)

CTXS = (512, 1024, 4096)


def test_bench():
    push_bundle_if_needed()
    run_adb_command(f"mkdir -p {MM_CACHE_PATH} {RESULTS_PATH} {PROMPTS_PATH}")

    subprocess.run(["adb", "push", HOST_IMAGE, IMAGE_PATH], check=True)
    subprocess.run(["adb", "push", f"{HOST_PROMPTS}/.", PROMPTS_PATH], check=True)

    chipset = Path(HOST_CHIPSET).read_text().strip()
    rows = [r for r in Path(HOST_ROWS).read_text().splitlines() if r.strip()]
    # Bucket cells by plugin: llama_cpp uses random-ids prefill (`-p N`),
    # qairt uses prompt_utf8 from a per-ctx text file (it doesn't accept
    # input_ids — see issue #1008).
    tsv_by_plugin_ctx: dict[tuple[str, int], list[str]] = {
        (plugin, ctx): [] for plugin in ("llama_cpp", "qairt") for ctx in CTXS
    }
    for row in rows:
        name, plugin, devs, model_id, vlm, image, *_spec = row.split("|")
        if plugin not in ("llama_cpp", "qairt"):
            continue
        if plugin != "qairt":
            vlm = image = ""
        imgpath = IMAGE_PATH if image == "1" else ""
        for d in devs.split(","):
            for ctx in CTXS:
                # Columns 5/6 (tokenizer/mmproj) intentionally blank: the
                # model manager fills both from the resolved manifest.
                tsv_by_plugin_ctx[(plugin, ctx)].append(
                    f"{name}-{plugin}-{d}-c{ctx}\t{plugin}\t{d}\t{model_id}"
                    f"\t\t\t{imgpath}\t{vlm}"
                )

    assert any(tsv_by_plugin_ctx.values()), "no model rows produced"

    lib = f"{BUNDLE_PATH}/lib"
    env = (
        f"LD_LIBRARY_PATH={lib}:{lib}/llama_cpp:{lib}/qairt "
        f"ADSP_LIBRARY_PATH={lib} "
        f"GENIEX_PLUGIN_PATH={lib}"
    )
    failures: list[tuple[str, int]] = []
    for (plugin, ctx), rows_for_cell in tsv_by_plugin_ctx.items():
        if not rows_for_cell:
            continue
        bucket = "llama" if plugin == "llama_cpp" else "qairt"
        tsv_path = f"/data/local/tmp/matrix-{bucket}-{ctx}.tsv"
        prompt_arg = (
            f"-p {ctx}"
            if plugin == "llama_cpp"
            else f"--prompt-file {PROMPTS_PATH}/sample_prompt_{ctx}.txt"
        )
        run_adb_command(
            "printf '%s\\n' "
            + " ".join(f"'{ln}'" for ln in rows_for_cell)
            + f" > {tsv_path}"
        )
        res = run_adb_command(
            f"cd {BUNDLE_PATH} && {env} ./bin/geniex-bench "
            f"--matrix-file {tsv_path} --output-json-dir {RESULTS_PATH} -r 3 "
            f"-c {ctx} {prompt_arg} "
            f"--mm-data-dir {MM_CACHE_PATH} --chipset '{chipset}'",
            check=False,
        )
        if res.returncode != 0:
            failures.append((plugin, ctx))
    listing = run_adb_command(f"ls {RESULTS_PATH}", check=False).stdout
    n_json = sum(1 for ln in listing.splitlines() if ln.strip().endswith(".json"))
    assert not failures, f"geniex-bench failed for {failures}"
    assert (
        n_json > 0
    ), f"no cell JSON produced on device (ls {RESULTS_PATH}: {listing!r})"


if __name__ == "__main__":
    import pytest

    raise SystemExit(
        pytest.main(["-s", "--junitxml=results.xml", os.path.realpath(__file__)])
    )
