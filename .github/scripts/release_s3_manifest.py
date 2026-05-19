#!/usr/bin/env python3
"""Generate S3 release-manifest JSON files for the geniex public bucket.

Three artifacts, all served anonymously from
``s3://qaihub-public-assets/qai-hub-geniex/``:

* ``manifest-<tag>.json`` — per-tag asset listing (immutable).
* ``index.json``          — full version catalogue (mutable, no-cache).
* ``latest.json``         — pointer to the latest stable tag (mutable, no-cache;
  only refreshed on bare ``vX.Y.Z`` tags).

The script only writes local files; the calling workflow uploads them with
``aws s3 cp`` so credentials, ACL flags, and cache headers stay in one place.
``index.json`` is fetched from S3 with ``aws s3api`` if it already exists.

Inputs come from environment variables to keep the workflow surface tight:

* ``RELEASE_TAG``        — e.g. ``v0.1.6-rc.1``.
* ``IS_PRERELEASE``      — ``true`` / ``false`` (matches the resolve-tag job).
* ``HTP_SIGNED``         — ``true`` / ``false`` from overlay-htp.
* ``LLAMA_SHA``          — short SHA from overlay-htp.
* ``RELEASE_ASSETS_DIR`` — directory containing the downloaded release-assets
  artifact (zips, exe, apk, sha256 sidecars).
* ``OUTPUT_DIR``         — where to write the generated JSON files.
* ``S3_BUCKET``          — default ``qaihub-public-assets``.
* ``S3_PREFIX``          — default ``qai-hub-geniex/`` (trailing slash kept).
* ``PUBLIC_BASE_URL``    — default
  ``https://qaihub-public-assets.s3.us-west-2.amazonaws.com``.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import subprocess
import sys
from pathlib import Path

SCHEMA_VERSION = 1

# Files that the workflow uploads to S3. Anything not in here is GitHub-Release-only.
ASSET_PATTERNS: list[tuple[re.Pattern[str], dict[str, str]]] = [
    (re.compile(r"^geniex-sdk-windows-arm64-.+\.zip$"),
     {"kind": "sdk", "platform": "windows", "arch": "arm64"}),
    (re.compile(r"^geniex-sdk-linux-arm64-.+\.zip$"),
     {"kind": "sdk", "platform": "linux", "arch": "arm64"}),
    (re.compile(r"^geniex-cli-setup-windows-arm64-.+\.exe$"),
     {"kind": "cli-installer", "platform": "windows", "arch": "arm64"}),
    (re.compile(r"^geniex-cli-linux-arm64-.+\.tar\.gz$"),
     {"kind": "cli-archive", "platform": "linux", "arch": "arm64"}),
    (re.compile(r"^install-.+\.sh$"),
     {"kind": "install-script", "platform": "linux", "arch": "arm64"}),
    (re.compile(r"^geniex-demo-.+\.apk$"),
     {"kind": "android-demo", "platform": "android", "arch": "arm64"}),
]


def now_utc_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def read_sidecar_sha256(sidecar: Path) -> str:
    # `sha256sum` writes "<hex>  <filename>".
    return sidecar.read_text().split()[0]


def classify(name: str) -> dict[str, str] | None:
    if name.endswith(".sha256"):
        base = name[: -len(".sha256")]
        meta = classify(base)
        if meta is None:
            return None
        return {**meta, "kind": "sha256"}
    for pat, meta in ASSET_PATTERNS:
        if pat.match(name):
            return dict(meta)
    return None


def build_tag_manifest(args: argparse.Namespace) -> None:
    tag = os.environ["RELEASE_TAG"]
    assets_dir = Path(os.environ["RELEASE_ASSETS_DIR"])
    output_dir = Path(os.environ["OUTPUT_DIR"])
    output_dir.mkdir(parents=True, exist_ok=True)
    base_url = os.environ.get(
        "PUBLIC_BASE_URL",
        "https://qaihub-public-assets.s3.us-west-2.amazonaws.com",
    ).rstrip("/")
    prefix = os.environ.get("S3_PREFIX", "qai-hub-geniex/").strip("/")
    bucket = os.environ.get("S3_BUCKET", "qaihub-public-assets")

    is_prerelease = os.environ["IS_PRERELEASE"].lower() == "true"
    htp_signed = os.environ.get("HTP_SIGNED", "").lower() == "true"
    llama_sha = os.environ.get("LLAMA_SHA", "")

    # Per-tag manifest is immutable. On a workflow re-run, preserve the original
    # released_at so the file stays byte-identical.
    prior = fetch_existing_object(bucket, f"{prefix}/manifest-{tag}.json")
    released_at = prior["released_at"] if prior else now_utc_iso()

    assets: list[dict[str, object]] = []
    for entry in sorted(assets_dir.iterdir()):
        if not entry.is_file():
            continue
        meta = classify(entry.name)
        if meta is None:
            continue
        sha_path = entry.with_name(entry.name + ".sha256")
        if entry.name.endswith(".sha256"):
            # Sidecar's own sha256 is the one inside its content's sidecar — i.e. itself.
            # Recompute on the fly to keep the manifest self-contained.
            import hashlib
            sha = hashlib.sha256(entry.read_bytes()).hexdigest()
        else:
            if not sha_path.exists():
                raise SystemExit(f"missing sha256 sidecar for {entry.name}")
            sha = read_sidecar_sha256(sha_path)
        assets.append({
            "name": entry.name,
            "url": f"{base_url}/{prefix}/{entry.name}",
            "size": entry.stat().st_size,
            "sha256": sha,
            **meta,
        })

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "tag": tag,
        "is_prerelease": is_prerelease,
        "released_at": released_at,
        "llama_sha": llama_sha,
        "htp_signed": htp_signed,
        "assets": assets,
    }
    out = output_dir / f"manifest-{tag}.json"
    out.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {out} ({len(assets)} assets)")


# SemVer 2.0 ordering, restricted to the v-prefix used by this repo.
_SEMVER_RE = re.compile(
    r"^v(?P<major>\d+)\.(?P<minor>\d+)\.(?P<patch>\d+)"
    r"(?:-(?P<pre>[0-9A-Za-z.-]+))?(?:\+[0-9A-Za-z.-]+)?$",
)


def semver_key(tag: str) -> tuple:
    """Sort key for SemVer tags. Pre-release sorts lower than its release."""
    m = _SEMVER_RE.match(tag)
    if not m:
        return (0, 0, 0, (1,), tag)
    base = (int(m["major"]), int(m["minor"]), int(m["patch"]))
    pre = m["pre"]
    if pre is None:
        # Release > any prerelease for the same base.
        return (*base, (1,), "")
    parts: list = []
    for ident in pre.split("."):
        parts.append((0, int(ident)) if ident.isdigit() else (1, ident))
    return (*base, (0, *parts), pre)


def fetch_existing_object(bucket: str, key: str) -> dict | None:
    """GET a JSON object from S3 via aws s3api, None if it doesn't exist."""
    import tempfile

    with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp:
        tmp_path = tmp.name
    try:
        proc = subprocess.run(
            ["aws", "s3api", "get-object", "--bucket", bucket, "--key", key, tmp_path],
            capture_output=True,
        )
        if proc.returncode != 0:
            stderr = proc.stderr.decode("utf-8", errors="replace")
            if "NoSuchKey" in stderr or "404" in stderr or "Not Found" in stderr:
                return None
            raise SystemExit(f"aws s3api get-object failed: {stderr}")
        return json.loads(Path(tmp_path).read_text())
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def fetch_existing_index(bucket: str, prefix: str) -> dict | None:
    return fetch_existing_object(bucket, f"{prefix.strip('/')}/index.json")


def update_index(args: argparse.Namespace) -> None:
    tag = os.environ["RELEASE_TAG"]
    is_prerelease = os.environ["IS_PRERELEASE"].lower() == "true"
    output_dir = Path(os.environ["OUTPUT_DIR"])
    bucket = os.environ.get("S3_BUCKET", "qaihub-public-assets")
    prefix = os.environ.get("S3_PREFIX", "qai-hub-geniex/")

    existing = fetch_existing_index(bucket, prefix) or {
        "schema_version": SCHEMA_VERSION,
        "versions": [],
    }
    prior = next(
        (v for v in existing.get("versions", []) if v.get("tag") == tag), None,
    )
    versions = [v for v in existing.get("versions", []) if v.get("tag") != tag]
    versions.append({
        "tag": tag,
        "is_prerelease": is_prerelease,
        # Preserve the original release timestamp on workflow re-runs of the
        # same tag — only set it on the first publish.
        "released_at": prior["released_at"] if prior else now_utc_iso(),
        "manifest": f"manifest-{tag}.json",
    })
    versions.sort(key=lambda v: semver_key(v["tag"]), reverse=True)

    latest_stable = next(
        (v["tag"] for v in versions if not v.get("is_prerelease")), None,
    )
    latest_prerelease = next(
        (v["tag"] for v in versions if v.get("is_prerelease")), None,
    )

    index = {
        "schema_version": SCHEMA_VERSION,
        "updated_at": now_utc_iso(),
        "latest_stable": latest_stable,
        "latest_prerelease": latest_prerelease,
        "versions": versions,
    }
    out = output_dir / "index.json"
    out.write_text(json.dumps(index, indent=2) + "\n")
    print(f"wrote {out} ({len(versions)} versions)")


def build_latest(args: argparse.Namespace) -> None:
    tag = os.environ["RELEASE_TAG"]
    if os.environ["IS_PRERELEASE"].lower() == "true":
        print("prerelease — skipping latest.json")
        return
    output_dir = Path(os.environ["OUTPUT_DIR"])
    # Carry the released_at from the just-built per-tag manifest so latest.json
    # stays consistent with it across re-runs.
    manifest_path = output_dir / f"manifest-{tag}.json"
    released_at = json.loads(manifest_path.read_text())["released_at"]
    out = output_dir / "latest.json"
    out.write_text(json.dumps({
        "schema_version": SCHEMA_VERSION,
        "tag": tag,
        "released_at": released_at,
        "manifest": f"manifest-{tag}.json",
    }, indent=2) + "\n")
    print(f"wrote {out}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build-tag-manifest").set_defaults(fn=build_tag_manifest)
    sub.add_parser("update-index").set_defaults(fn=update_index)
    sub.add_parser("build-latest").set_defaults(fn=build_latest)
    args = parser.parse_args()
    args.fn(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
