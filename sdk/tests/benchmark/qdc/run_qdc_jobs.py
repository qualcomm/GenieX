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
GITHUB_STEP_SUMMARY. Linux (QCS9075M) is implemented; the platform seam keeps
android/windows as future drop-ins.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import tempfile
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

# The QDC SDK is only needed in run mode; render mode (the aggregate job) has no
# wheel installed, so import it optionally and fail loudly only when run mode uses it.
try:
    from qualcomm_device_cloud_sdk.api import qdc_api
    from qualcomm_device_cloud_sdk.models import (
        ArtifactType,
        JobMode,
        JobState,
        JobSubmissionParameter,
        JobType,
        TestFramework,
    )
except ImportError:
    qdc_api = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

POLL_INTERVAL = 30
LOG_UPLOAD_TIMEOUT = 600
HERE = Path(__file__).parent


def platform_for(device: str) -> str:
    if device.startswith("QCS"):
        return "linux"
    if device.startswith("SM"):
        return "android"
    if device.startswith(("SC", "CRD", "X")):
        return "windows"
    raise SystemExit(f"unknown device chipset: {device}")


def build_linux_artifact(pkg_dir: Path, models: list[dict], tmp: Path) -> Path:
    stage = tmp / "stage"
    shutil.copytree(pkg_dir, stage / "pkg-geniex")

    rows = []
    for m in models:
        kind = "bundle" if "bundle_url" in m else "gguf"
        url = m.get("bundle_url") or m["url"]
        rows.append(f"{m['name']}|{m['plugin']}|{','.join(m['devices'])}|{url}|{kind}")
    script = (
        (HERE / "linux" / "run_linux.sh")
        .read_text()
        .replace("{MODELS}", "\n".join(rows))
    )

    script_path = stage / "run_linux.sh"
    script_path.write_text(script, newline="\n")
    script_path.chmod(0o755)

    return Path(shutil.make_archive(str(tmp / "artifact"), "zip", stage))


ENTRY = {"linux": "/bin/bash /data/local/tmp/TestContent/run_linux.sh"}
BUILDERS = {"linux": build_linux_artifact}


def wait_for_job(client, job_id: str, timeout: int) -> str:
    terminal = {JobState.COMPLETED, JobState.CANCELED}
    elapsed = 0
    while elapsed < timeout:
        raw = qdc_api.get_job_status(client, job_id)
        try:
            state = JobState(raw)
        except ValueError:
            state = None
        if state in terminal:
            return raw.lower()
        log.info("job %s: %s", job_id, raw)
        time.sleep(POLL_INTERVAL)
        elapsed += POLL_INTERVAL
    qdc_api.abort_job(client, job_id)
    raise TimeoutError(f"job {job_id} did not finish within {timeout}s")


def download_cells(client, job_id: str, tmp: Path) -> list[dict]:
    elapsed = 0
    while elapsed < LOG_UPLOAD_TIMEOUT:
        status = (qdc_api.get_job_log_upload_status(client, job_id) or "").lower()
        if status in {"completed", "failed"}:
            break
        log.info("waiting for log upload (status=%s)", status)
        time.sleep(POLL_INTERVAL)
        elapsed += POLL_INTERVAL

    cells = []
    for lf in qdc_api.get_job_log_files(client, job_id) or []:
        if "results/" not in lf.filename or not lf.filename.endswith(".json"):
            continue
        zip_path = tmp / "log.zip"
        qdc_api.download_job_log_files(client, lf.filename, str(zip_path))
        with zipfile.ZipFile(zip_path) as z:
            for name in z.namelist():
                if name.endswith(".json"):
                    cells.append(json.loads(z.read(name)))
    return sorted(cells, key=lambda c: c["cell_id"])


def _fmt(agg: dict, key: str) -> str:
    v = agg.get(key, {}).get("median")
    return f"{v:.1f}" if v else "-"


def render(cells: list[dict], device: str, tag: str, sha: str) -> str:
    lines = [
        f"## QDC Scorecard — {device} — {tag}",
        "",
        f"- geniex version: `{tag}`",
        f"- git sha: `{sha}`",
        f"- generated: `{datetime.now(timezone.utc).isoformat(timespec='seconds')}`",
        "",
        "| Model | Backend | Device | TTFT (ms) | Prefill (tok/s) | Decode (tok/s) | Gen tokens |",
        "|-------|---------|--------|-----------|------------------|-----------------|------------|",
    ]
    for c in sorted(cells, key=lambda c: c["cell_id"]):
        agg = c.get("agg", {})
        model = c["cell_id"].removesuffix(f"-{c['plugin']}-{c['device']}")
        gen = agg.get("gen_tokens", {}).get("median")
        gen = int(gen) if gen else "-"
        lines.append(
            f"| {model} | {c['plugin']} | "
            f"{c['device']} | {_fmt(agg, 'ttft_ms')} | {_fmt(agg, 'prefill_tps')} | "
            f"{_fmt(agg, 'decode_tps')} | {gen} |"
        )
    return "\n".join(lines) + "\n"


def write_summary(text: str) -> None:
    print(text)
    if path := os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(path, "a") as f:
            f.write(text)


def render_aggregate(cells_dir: Path, device: str) -> int:
    cells = (
        [
            c
            for f in sorted(cells_dir.rglob("*.json"))
            for c in json.loads(f.read_text())
        ]
        if cells_dir.exists()
        else []
    )
    tag = os.environ.get("GENIEX_RELEASE_TAG") or "unknown"
    sha = (
        subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True
        ).stdout.strip()
        or "unknown"
    )
    if not cells:
        write_summary(f"## QDC Scorecard — {device} — {tag}\n\nNo results recovered.\n")
        return 0
    write_summary(render(cells, device, tag, sha))
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
        return render_aggregate(args.render_dir, args.device)

    if qdc_api is None:
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
    client = qdc_api.get_public_api_client_using_api_key(
        api_key_header=api_key,
        app_name_header="geniex-ci",
        on_behalf_of_header="geniex-ci",
        client_type_header="Python",
    )
    target_id = qdc_api.get_target_id(client, args.device)
    if target_id is None:
        raise SystemExit(f"no QDC target for {args.device}")

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        zip_path = BUILDERS[platform](args.pkg_dir, models, tmp)
        log.info("uploading artifact (%d MB)", zip_path.stat().st_size // 1_000_000)
        artifact_id = qdc_api.upload_file(
            client, str(zip_path), ArtifactType.TESTSCRIPT
        )

        job_id = qdc_api.submit_job(
            public_api_client=client,
            target_id=target_id,
            job_name=f"geniex-bench-{args.device}"[:32],
            external_job_id=None,
            job_type=JobType.AUTOMATED,
            job_mode=JobMode.APPLICATION,
            timeout=max(1, args.job_timeout // 60),
            test_framework=TestFramework.BASH,
            entry_script=ENTRY[platform],
            job_artifacts=[artifact_id],
            monkey_events=None,
            monkey_session_timeout=None,
            job_parameters=[JobSubmissionParameter.WIFIENABLED],
        )
        log.info("job submitted: %s (device=%s)", job_id, args.device)
        status = wait_for_job(client, job_id, args.job_timeout)
        log.info("job %s finished: %s", job_id, status)
        cells = download_cells(client, job_id, tmp)

    if args.cells_out:
        args.cells_out.write_text(json.dumps(cells))

    if not cells:
        raise SystemExit("no benchmark results recovered from the device")

    # Render this model's own table into its job summary for immediate visibility;
    # the aggregate job later flattens every model's cells into one unified table.
    tag = os.environ.get("GENIEX_RELEASE_TAG") or "unknown"
    sha = (
        subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True
        ).stdout.strip()
        or "unknown"
    )
    write_summary(render(cells, args.device, tag, sha))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
