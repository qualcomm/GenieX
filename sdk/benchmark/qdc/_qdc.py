# Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Shared QDC primitives: client, job submit/poll, and log download.

Both the benchmark runner (run_qdc_jobs.py) and the pytest harness
(tests/qdc/run_qdc_pytest.py) submit an artifact, poll until the job is
terminal, then pull files out of the device's QDC_logs. The QDC-specific
plumbing lives here so neither caller reimplements it.
"""

from __future__ import annotations

import json
import logging
import random
import re
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
SUBMIT_RETRY_BUDGET = 7200
SUBMIT_BACKOFF_BASE = 30
SUBMIT_BACKOFF_CAP = 300
TRANSIENT_RETRY_ATTEMPTS = 5
TRANSIENT_BACKOFF_BASE = 10
TRANSIENT_BACKOFF_CAP = 60

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


# QDC's upload/status/log endpoints occasionally blip with a 5xx (mostly 504
# Gateway Time-out, some 500) with no fault on our side; retrying the same
# call a few seconds later almost always succeeds. Unlike the pending-job
# quota, this isn't capacity we need to wait out, so the retry is short.
# A 504's body is an empty/HTML gateway page rather than JSON, and the SDK's
# try_call() does a bare json.loads() on it without checking the status code
# first, so the observable symptom is often a JSONDecodeError rather than an
# exception that mentions "status code 5xx".
_TRANSIENT_STATUS_RE = re.compile(r"status code 5\d\d\b")


def _is_transient_error(exc: Exception) -> bool:
    if isinstance(exc, json.JSONDecodeError):
        return True
    return bool(_TRANSIENT_STATUS_RE.search(str(exc)))


def _call_with_retry(fn, *args, what: str, **kwargs):
    attempt = 0
    while True:
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            if not _is_transient_error(exc) or attempt >= TRANSIENT_RETRY_ATTEMPTS:
                raise
            sleep = min(TRANSIENT_BACKOFF_CAP, TRANSIENT_BACKOFF_BASE * 2**attempt)
            log.warning(
                "%s hit a transient error (attempt %d): %s; retrying in %ds",
                what,
                attempt + 1,
                exc,
                sleep,
            )
            time.sleep(sleep)
            attempt += 1


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
        raw = _call_with_retry(
            qdc_api.get_job_status, client, job_id, what="job status poll"
        )
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
    # in lockstep) until a slot frees up or the budget runs out. Under a full
    # matrix run the quota realistically stays contested for closer to two
    # hours than one (observed elapsed-before-giveup times up to ~54min on a
    # 60min budget, with the "N pending" count still fluctuating rather than
    # stuck), so the budget leaves comfortable room under the job's timeout.
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
    artifact_id = _call_with_retry(
        qdc_api.upload_file,
        client,
        str(zip_path),
        ArtifactType.TESTSCRIPT,
        what="upload artifact",
    )
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


def _basename(name: str) -> str:
    return name.replace("\\", "/").rsplit("/", 1)[-1]


def download_log_members(
    client, job_id: str, tmp: Path, want: Callable[[str], bool]
) -> list[tuple[str, bytes]]:
    """Return (member_name, bytes) for every collected log member matching ``want``.

    ``want`` is invoked on the *basename* of each candidate — both the QDC
    log-file name (path-shaped, e.g. ``.../results/cell.json``) and any inner
    zip member (already a basename) — so callers filter on extension / prefix
    without caring whether QDC ships the file raw or zipped.
    """
    elapsed = 0
    while elapsed < LOG_UPLOAD_TIMEOUT:
        status = (
            _call_with_retry(
                qdc_api.get_job_log_upload_status,
                client,
                job_id,
                what="log upload status poll",
            )
            or ""
        ).lower()
        if status in {"completed", "failed"}:
            break
        log.info("waiting for log upload (status=%s)", status)
        time.sleep(POLL_INTERVAL)
        elapsed += POLL_INTERVAL

    out: list[tuple[str, bytes]] = []
    log_files = _call_with_retry(
        qdc_api.get_job_log_files, client, job_id, what="list job log files"
    )
    for lf in log_files or []:
        if not want(_basename(lf.filename)):
            continue
        dl = tmp / "log.bin"
        _call_with_retry(
            qdc_api.download_job_log_files,
            client,
            lf.filename,
            str(dl),
            what="download job log file",
        )
        if zipfile.is_zipfile(dl):
            with zipfile.ZipFile(dl) as z:
                for name in z.namelist():
                    if want(_basename(name)):
                        out.append((name, z.read(name)))
        else:
            out.append((lf.filename, dl.read_bytes()))
    return out
