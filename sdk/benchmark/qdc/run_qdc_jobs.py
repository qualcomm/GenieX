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

"""Run geniex_benchmark on a QDC device and render a scorecard.

Builds an artifact (SDK pkg + entry script), submits it as a QDC job, downloads
the per-cell JSON geniex_benchmark emits, and writes a markdown scorecard to
GITHUB_STEP_SUMMARY. Linux (QCS9075M, BASH), Windows (SC8380XP, PowerShell), and
Android (SM8850, APPIUM via adb) are implemented.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

# The QDC SDK is only needed in run mode; render mode (the aggregate job) has no
# wheel installed, so import the shared primitives optionally and fail loudly
# only when run mode uses them.
try:
    import _qdc
except ImportError:
    _qdc = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

HERE = Path(__file__).parent

AIHUB_BASE = "https://qaihub-public-assets.s3.us-west-2.amazonaws.com/qai-hub-models"
AIHUB_VERSION = "v0.52.0"
# Staged into every artifact and fed to VLM cells; reuses the committed VLM
# e2e fixture (tests/conftest.py TEST_IMAGE_PATH).
TEST_IMAGE = HERE.parents[2] / "cli" / "server" / "docs" / "ui" / "favicon-32x32.png"
CHIPSET = {
    "QCS9075M": "qualcomm-qcs9075",
    "SC8380XP": "qualcomm-snapdragon-x-elite",
    "SC8480XP": "qualcomm-snapdragon-x2-elite",
    "SM8750": "qualcomm-snapdragon-8-elite",
    "SM8850": "qualcomm-snapdragon-8-elite-gen5",
}


def platform_for(device: str) -> str:
    if device.startswith("QCS"):
        return "linux"
    if device.startswith("SM"):
        return "android"
    if device.startswith(("SC", "CRD", "X")):
        return "windows"
    raise SystemExit(f"unknown device chipset: {device}")


def resolve_bundle_url(aihub_id: str, device: str) -> str | None:
    slug = CHIPSET.get(device)
    if slug is None:
        raise SystemExit(f"no chipset slug for {device}")
    url = f"{AIHUB_BASE}/models/{aihub_id}/releases/{AIHUB_VERSION}/release-assets.json"
    with urllib.request.urlopen(url) as r:
        assets = json.load(r).get("assets", [])
    return next((a["download_url"] for a in assets if a.get("chipset") == slug), None)


def resolve_model_url(m: dict, device: str) -> tuple[str | None, str]:
    """Return (download_url, kind) for a model entry; url is None when no
    asset matches the device (QAIRT bundles only)."""
    if "aihub_id" in m:
        return resolve_bundle_url(m["aihub_id"], device), "bundle"
    return m["url"], "gguf"


def model_rows(models: list[dict], device: str) -> list[str]:
    rows = []
    for m in models:
        url, kind = resolve_model_url(m, device)
        if url is None:
            log.warning("no %s asset for %s, skipping", device, m["name"])
            continue
        mmproj = m.get("mmproj_url", "")
        vlm = "1" if m.get("vlm") else ""
        image = "1" if m.get("image") else ""
        rows.append(
            f"{m['name']}|{m['plugin']}|{','.join(m['devices'])}|{url}|{kind}"
            f"|{mmproj}|{vlm}|{image}"
        )
    return rows


def build_linux_artifact(
    pkg_dir: Path, models: list[dict], device: str, tmp: Path
) -> Path:
    stage = tmp / "stage"
    shutil.copytree(pkg_dir, stage / "pkg-geniex")

    script = (
        (HERE / "linux" / "run_linux.sh")
        .read_text()
        .replace("{MODELS}", "\n".join(model_rows(models, device)))
    )
    script_path = stage / "run_linux.sh"
    script_path.write_text(script, newline="\n")
    script_path.chmod(0o755)

    shutil.copy(TEST_IMAGE, stage / "test.png")
    shutil.copytree(HERE / "prompts", stage / "prompts")

    return Path(shutil.make_archive(str(tmp / "artifact"), "zip", stage))


def build_windows_artifact(
    pkg_dir: Path, models: list[dict], device: str, tmp: Path
) -> Path:
    stage = tmp / "stage"
    shutil.copytree(pkg_dir, stage / "pkg-geniex")

    script = (
        (HERE / "windows" / "run_windows.ps1")
        .read_text()
        .replace("{MODELS}", "\n".join(model_rows(models, device)))
    )
    (stage / "run_windows.ps1").write_text(script, newline="\r\n")

    cert = HERE.parents[2] / ".github" / "certs" / "hexagon" / "ggml-htp-v1.cer"
    shutil.copy(cert, stage / "ggml-htp-v1.cer")

    shutil.copy(TEST_IMAGE, stage / "test.png")
    shutil.copytree(HERE / "prompts", stage / "prompts")

    return Path(shutil.make_archive(str(tmp / "artifact"), "zip", stage))


def build_android_artifact(
    pkg_dir: Path, models: list[dict], device: str, tmp: Path
) -> Path:
    # Phones lack python3/curl, so the appium pytest harness on the QDC host
    # fetches+extracts each model and adb-pushes it, then runs geniex_benchmark
    # on-device; results land in the device's QDC_logs and are auto-collected.
    stage = tmp / "stage"
    shutil.copytree(pkg_dir, stage / "pkg-geniex")
    # libggml-cpu.so needs libomp.so, which the CLI package doesn't ship but the
    # Android app provides from extLibs; drop it beside the ggml libs.
    omp = (
        HERE.parents[2]
        / "bindings"
        / "android"
        / "app"
        / "extLibs"
        / "arm64-v8a"
        / "libomp.so"
    )
    shutil.copy(omp, stage / "pkg-geniex" / "lib" / "llama_cpp" / "libomp.so")
    (stage / "matrix_rows.txt").write_text("\n".join(model_rows(models, device)))
    shutil.copytree(HERE / "tests", stage / "tests")
    shutil.copy(HERE / "tests" / "requirements.txt", stage / "requirements.txt")
    shutil.copy(TEST_IMAGE, stage / "test.png")
    shutil.copytree(HERE / "prompts", stage / "prompts")
    (stage / "pytest.ini").write_text("[pytest]\naddopts = --junitxml=results.xml\n")

    return Path(shutil.make_archive(str(tmp / "artifact"), "zip", stage))


ENTRY = {
    "linux": "/bin/bash /data/local/tmp/TestContent/run_linux.sh",
    "windows": "C:\\Temp\\TestContent\\run_windows.ps1",
    "android": None,
}
BUILDERS = {
    "linux": build_linux_artifact,
    "windows": build_windows_artifact,
    "android": build_android_artifact,
}


def download_cells(client, job_id: str, tmp: Path) -> list[dict]:
    members = _qdc.download_log_members(
        client, job_id, tmp, lambda n: n.endswith(".json")
    )
    cells = [json.loads(data) for _, data in members]
    return sorted(cells, key=lambda c: c["cell_id"])


def _fmt_med_sd(agg: dict, key: str) -> str:
    entry = agg.get(key) or {}
    med = entry.get("median")
    sd = entry.get("stdev")
    if med is None:
        return "-"
    if sd is None:
        return f"{med:.1f}"
    return f"{med:.1f} ± {sd:.1f}"


LLAMA_CPP_COMMIT_BASE = "https://github.com/ggml-org/llama.cpp/commit"


_CTX_SUFFIX = re.compile(r"-c(\d+)$")


def _ctx_from_cell(c: dict) -> int:
    """Pull ctx from the `-c{N}` cell_id suffix; fall back to params.n_ctx."""
    m = _CTX_SUFFIX.search(c.get("cell_id") or "")
    if m:
        return int(m.group(1))
    return int((c.get("params") or {}).get("n_ctx") or 0)


def _model_label(c: dict) -> str:
    cid = _CTX_SUFFIX.sub("", c.get("cell_id") or "")
    return cid.removesuffix(f"-{c['plugin']}-{c['device']}")


def detect_geniex_label(sha: str) -> str:
    """Return `<tag> (<sha>)` when the current commit is a release tag, else
    just `<sha>`. Three-layer fallback: explicit env > exact-match git tag >
    bare sha."""
    if env := os.environ.get("GENIEX_RELEASE_TAG"):
        return f"{env} ({sha})"
    r = subprocess.run(
        ["git", "describe", "--tags", "--exact-match", "HEAD"],
        capture_output=True,
        text=True,
    )
    if r.returncode == 0 and (tag := r.stdout.strip()):
        return f"{tag} ({sha})"
    return sha or "unknown"


def _pick_field(cells: list[dict], key: str) -> str | None:
    return next((v for c in cells if (v := c.get(key))), None)


def _models_block(models: list[dict], device: str) -> list[str]:
    lines = ["**Models**", ""]
    for m in models:
        url, _ = resolve_model_url(m, device)
        lines.append(f"- {m['name']}: {url or '-'}")
    return lines


def _details_block(
    cells: list[dict], device: str, label: str, models: list[dict] | None
) -> list[str]:
    qairt_v = _pick_field(cells, "qairt_version") or "-"
    llama_v = _pick_field(cells, "llama_cpp_version")
    llama_line = f"[`{llama_v}`]({LLAMA_CPP_COMMIT_BASE}/{llama_v})" if llama_v else "-"
    lines = [
        "<details><summary>Build & models</summary>",
        "",
        "**Versions**",
        "",
        f"- geniex: `{label}`",
        f"- QAIRT: `{qairt_v}`",
        f"- llama.cpp: {llama_line}",
        f"- generated: `{datetime.now(timezone.utc).isoformat(timespec='seconds')}`",
        "",
    ]
    if models:
        lines += _models_block(models, device)
        lines.append("")
    lines += ["</details>", ""]
    return lines


def render(
    cells: list[dict],
    device: str,
    label: str,
    models: list[dict] | None = None,
) -> str:
    lines = [f"## QDC Scorecard — {device} — {label}", ""]
    lines += _details_block(cells, device, label, models)
    lines += [
        "| Model | Backend | Device | Ctx | ngl | Test | TTFT (ms) | Prefill (tok/s) | Decode (tok/s) |",
        "|-------|---------|--------|----:|----:|------|----------:|----------------:|---------------:|",
    ]
    sort_key = lambda c: (_model_label(c), c["plugin"], c["device"], _ctx_from_cell(c))  # noqa: E731
    for c in sorted(cells, key=sort_key):
        agg = c.get("agg") or {}
        params = c.get("params") or {}
        model = _model_label(c)
        ngl_v = params.get("n_gpu_layers")
        ngl = "-" if c["plugin"] == "qairt" or not ngl_v else str(ngl_v)
        ctx = _ctx_from_cell(c)
        ctx_s = str(ctx) if ctx else "-"
        p_med = (agg.get("prompt_tokens") or {}).get("median")
        g_med = (agg.get("gen_tokens") or {}).get("median")
        test = (
            f"pp{int(p_med)}+tg{int(g_med)}"
            if p_med is not None and g_med is not None
            else "-"
        )
        lines.append(
            f"| {model} | {c['plugin']} | {c['device']} | {ctx_s} | {ngl} | {test} | "
            f"{_fmt_med_sd(agg, 'ttft_ms')} | {_fmt_med_sd(agg, 'prefill_tps')} | "
            f"{_fmt_med_sd(agg, 'decode_tps')} |"
        )
    return "\n".join(lines) + "\n"


def write_summary(text: str) -> None:
    print(text)
    if path := os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(path, "a") as f:
            f.write(text)


def _short_sha() -> str:
    return (
        subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True
        ).stdout.strip()
        or "unknown"
    )


def render_aggregate(cells_dir: Path, device: str, models_file: Path) -> int:
    cells = (
        [
            c
            for f in sorted(cells_dir.rglob("*.json"))
            for c in json.loads(f.read_text())
        ]
        if cells_dir.exists()
        else []
    )
    label = detect_geniex_label(_short_sha())
    if not cells:
        write_summary(
            f"## QDC Scorecard — {device} — {label}\n\nNo results recovered.\n"
        )
        return 0
    models = json.loads(models_file.read_text()) if models_file.exists() else None
    write_summary(render(cells, device, label, models))
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--pkg-dir", type=Path)
    p.add_argument("--device", default="QCS9075M")
    p.add_argument("--models-file", type=Path, default=HERE / "scorecard-models.json")
    p.add_argument("--model-name", help="run only this model from --models-file")
    p.add_argument("--cells-out", type=Path, help="write the per-cell JSON list here")
    p.add_argument("--render-dir", type=Path, help="render mode: aggregate JSON here")
    p.add_argument("--job-timeout", type=int, default=7200)
    args = p.parse_args()

    if args.render_dir:
        return render_aggregate(args.render_dir, args.device, args.models_file)

    if _qdc is None:
        raise SystemExit("qualcomm_device_cloud_sdk is required for run mode")
    api_key = os.environ.get("QDC_API_KEY")
    if not api_key:
        raise SystemExit("QDC_API_KEY must be set")
    if not args.pkg_dir:
        raise SystemExit("--pkg-dir is required")

    platform = platform_for(args.device)
    if platform not in BUILDERS:
        raise SystemExit(f"{platform} not implemented yet")

    models = json.loads(args.models_file.read_text())
    if args.model_name:
        models = [m for m in models if m["name"] == args.model_name]
        if not models:
            raise SystemExit(f"model {args.model_name!r} not in {args.models_file}")
    client = _qdc.make_client(api_key)
    target_id = _qdc.resolve_target(client, args.device)

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        zip_path = BUILDERS[platform](args.pkg_dir, models, args.device, tmp)
        job_id = _qdc.submit_and_wait(
            client,
            target_id=target_id,
            job_name=f"geniex-bench-{args.device}",
            platform=platform,
            entry_script=ENTRY[platform],
            zip_path=zip_path,
            timeout=args.job_timeout,
        )
        cells = download_cells(client, job_id, tmp)

    if args.cells_out:
        args.cells_out.write_text(json.dumps(cells))

    if not cells:
        raise SystemExit("no benchmark results recovered from the device")

    # Render this model's own table into its job summary for immediate visibility;
    # the aggregate job later flattens every model's cells into one unified table.
    write_summary(render(cells, args.device, detect_geniex_label(_short_sha()), models))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
