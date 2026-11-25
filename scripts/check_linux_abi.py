#!/usr/bin/env python3
# Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Check the ABI requirements of packaged ARM64 Linux host binaries."""

import argparse
import os
from pathlib import Path
import re
import subprocess
import sys


LIMITS = {"GLIBC": (2, 41), "GLIBCXX": (3, 4, 30), "CXXABI": (1, 3, 13)}
VERSION = re.compile(r"\bName: (GLIBCXX|GLIBC|CXXABI)_([0-9]+(?:\.[0-9]+)*)\b")


def readelf(option, path):
    return subprocess.run(
        ["readelf", "--wide", option, str(path)],
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "LC_ALL": "C"},
    ).stdout


def required_versions(output):
    in_needs = False
    for line in output.splitlines():
        if line.startswith("Version "):
            in_needs = line.startswith("Version needs section")
        if in_needs:
            match = VERSION.search(line)
            if match:
                yield match.group(1), tuple(map(int, match.group(2).split(".")))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="SDK directory or an ELF file")
    args = parser.parse_args()
    if not args.path.exists():
        parser.error(f"path does not exist: {args.path}")

    files = [args.path] if args.path.is_file() else sorted(args.path.rglob("*"))
    checked = 0
    errors = []
    try:
        for path in files:
            if not path.is_file():
                continue
            with path.open("rb") as stream:
                if stream.read(4) != b"\x7fELF":
                    continue
            if not re.search(
                r"Machine:\s+AArch64\s*$", readelf("--file-header", path), re.M
            ):
                continue
            checked += 1
            for family, version in required_versions(readelf("--version-info", path)):
                limit = LIMITS[family]
                if version > limit:
                    required = ".".join(map(str, version))
                    supported = ".".join(map(str, limit))
                    errors.append(
                        f"{path}: requires {family}_{required}; limit is {family}_{supported}"
                    )
    except (OSError, subprocess.CalledProcessError) as exc:
        print(f"error: cannot inspect Linux ABI: {exc}", file=sys.stderr)
        return 1

    if not checked:
        errors.append(f"{args.path}: no ARM64 Linux ELF files found")
    for error in errors:
        print(f"error: {error}", file=sys.stderr)
    if errors:
        return 1
    print(
        f"Linux ABI check passed for {checked} files (GLIBC 2.41 / GLIBCXX 3.4.30 / CXXABI 1.3.13)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
