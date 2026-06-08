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

"""Shared QDC primitives: client, job submit/poll, and log download.

Both the benchmark scorecard (run_qdc_jobs.py) and the pytest harness
(tests/qdc/run_qdc_pytest.py) submit an artifact, poll until the job is
terminal, then pull files out of the device's QDC_logs. The QDC-specific
plumbing lives here so neither caller reimplements it.
"""

from __future__ import annotations

import logging
import random
import time
import zipfile
from pathlib import Path
from typing import Callable

from qualcomm_device_cloud_sdk.api import qdc_api
from qualcomm_device_cloud_sdk.models import (
    ArtifactType,
    JobMode,
    JobState,
    JobSubmissionParameter,
    JobType,
    TestFramework,
)

log = logging.getLogger(__name__)

POLL_INTERVAL = 30
LOG_UPLOAD_TIMEOUT = 600
SUBMIT_RETRY_BUDGET = 3600
SUBMIT_BACKOFF_BASE = 30
SUBMIT_BACKOFF_CAP = 300

FRAMEWORK = {
    "linux": TestFramework.BASH,
    "windows": TestFramework.POWERSHELL,
    "android": TestFramework.APPIUM,
}

# A QDC key allows a fixed number of pending jobs; over it, submit returns
# 400 "User <x> already has N pending jobs". Match that so we back off instead
# of crashing; the other hints cover adjacent capacity/quota phrasings.
_QUOTA_HINTS = ("pending jobs", "too many", "quota", "limit", "capacity")


def _is_quota_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(h in msg for h in _QUOTA_HINTS)


def make_client(api_key: str):
    return qdc_api.get_public_api_client_using_api_key(
        api_key_header=api_key,
        app_name_header="geniex-ci",
        on_behalf_of_header="geniex-ci",
        client_type_header="Python",
    )


def resolve_target(client, device: str):
    target_id = qdc_api.get_target_id(client, device)
    if target_id is None:
        raise SystemExit(f"no QDC target for {device}")
    return target_id


def _wait_for_job(client, job_id: str, timeout: int) -> str:
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


def _submit_with_retry(client, **submit_kwargs) -> str:
    # All max-parallel runners share the key's pending-job quota; instead of
    # crashing when it's full, back off (with jitter so the runners don't retry
    # in lockstep) until a slot frees up or the budget runs out.
    elapsed = 0
    attempt = 0
    while True:
        try:
            return qdc_api.submit_job(public_api_client=client, **submit_kwargs)
        except Exception as exc:
            if not _is_quota_error(exc):
                raise
            base = min(SUBMIT_BACKOFF_CAP, SUBMIT_BACKOFF_BASE * 2**attempt)
            sleep = base + random.uniform(0, base)
            if elapsed + sleep > SUBMIT_RETRY_BUDGET:
                raise
            log.warning(
                "submit hit pending-job quota (attempt %d, elapsed %ds): %s; retrying in %.0fs",
                attempt + 1,
                elapsed,
                exc,
                sleep,
            )
            time.sleep(sleep)
            elapsed += sleep
            attempt += 1


def submit_and_wait(
    client,
    *,
    target_id,
    job_name: str,
    platform: str,
    entry_script: str | None,
    zip_path: Path,
    timeout: int,
) -> str:
    """Upload the artifact, submit the job (retrying on quota), and block until terminal."""
    log.info("uploading artifact (%d MB)", zip_path.stat().st_size // 1_000_000)
    artifact_id = qdc_api.upload_file(client, str(zip_path), ArtifactType.TESTSCRIPT)
    job_id = _submit_with_retry(
        client,
        target_id=target_id,
        job_name=job_name[:32],
        external_job_id=None,
        job_type=JobType.AUTOMATED,
        job_mode=JobMode.APPLICATION,
        timeout=max(1, timeout // 60),
        test_framework=FRAMEWORK[platform],
        entry_script=entry_script,
        job_artifacts=[artifact_id],
        monkey_events=None,
        monkey_session_timeout=None,
        job_parameters=[JobSubmissionParameter.WIFIENABLED],
    )
    log.info("job submitted: %s", job_id)
    status = _wait_for_job(client, job_id, timeout)
    log.info("job %s finished: %s", job_id, status)
    return job_id


def download_log_members(
    client, job_id: str, tmp: Path, want: Callable[[str], bool]
) -> list[tuple[str, bytes]]:
    """Return (member_name, bytes) for every collected log member matching ``want``.

    ``want`` is applied to the QDC log-file name; if the downloaded file is a
    zip it is also applied to each member inside, so callers filter on
    extension / prefix without caring whether QDC ships the file raw or zipped.
    """
    elapsed = 0
    while elapsed < LOG_UPLOAD_TIMEOUT:
        status = (qdc_api.get_job_log_upload_status(client, job_id) or "").lower()
        if status in {"completed", "failed"}:
            break
        log.info("waiting for log upload (status=%s)", status)
        time.sleep(POLL_INTERVAL)
        elapsed += POLL_INTERVAL

    out: list[tuple[str, bytes]] = []
    for lf in qdc_api.get_job_log_files(client, job_id) or []:
        if not want(lf.filename):
            continue
        dl = tmp / "log.bin"
        qdc_api.download_job_log_files(client, lf.filename, str(dl))
        if zipfile.is_zipfile(dl):
            with zipfile.ZipFile(dl) as z:
                for name in z.namelist():
                    if want(name):
                        out.append((name, z.read(name)))
        else:
            out.append((lf.filename, dl.read_bytes()))
    return out
