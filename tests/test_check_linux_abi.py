# Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

import contextlib
import importlib.util
import io
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location(
    'check_linux_abi', Path(__file__).resolve().parents[1] / 'scripts' / 'check_linux_abi.py'
)
abi = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(abi)


class LinuxAbiTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.versions = {}
        self.machines = {}

    def elf(self, name, versions=(), machine='AArch64'):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'\x7fELFfixture')
        self.versions[path] = versions
        self.machines[path] = machine
        return path

    def readelf(self, option, path):
        if option == '--file-header':
            return f'  Machine: {self.machines[path]}\n'
        return (
            "Version needs section '.gnu.version_r' contains 1 entry:\n"
            + '\n'.join(f'  Name: {version}' for version in self.versions[path])
            + "\nVersion definition section '.gnu.version_d' contains 1 entry:\n"
            + '  Name: GLIBC_9.99\n'
        )

    def check(self, path=None, readelf=None):
        output = io.StringIO()
        with (
            patch.object(abi, 'readelf', side_effect=readelf or self.readelf),
            patch.object(sys, 'argv', ['check_linux_abi.py', str(path or self.root)]),
            contextlib.redirect_stdout(output),
            contextlib.redirect_stderr(output),
        ):
            result = abi.main()
        return result, output.getvalue()

    def test_benchmark_requirement_fails_package_audit(self):
        self.elf('lib/libgeniex.so', ['GLIBC_2.17'])
        benchmark = self.elf('bin/geniex-bench', ['GLIBC_2.42'])
        result, output = self.check()
        self.assertEqual(result, 1)
        self.assertIn(f'{benchmark}: requires GLIBC_2.42', output)

    def test_compatible_package_passes(self):
        self.elf('lib/libgeniex.so', ['GLIBC_2.41'])
        self.elf('bin/geniex-bench', ['GLIBCXX_3.4.30', 'CXXABI_1.3.13'])
        result, output = self.check()
        self.assertEqual(result, 0)
        self.assertIn('passed for 2 files', output)

    def test_non_elf_payload_and_dsp_are_skipped(self):
        self.elf('lib/libgeniex.so', ['GLIBC_2.17'])
        self.elf('lib/llama_cpp/libggml-htp-v73.so', machine='Qualcomm Hexagon')
        (self.root / 'README').write_text('SDK package')
        result, output = self.check()
        self.assertEqual(result, 0)
        self.assertIn('passed for 1 files', output)

    def test_dsp_only_package_fails(self):
        self.elf('lib/libggml-htp-v73.so', machine='Qualcomm Hexagon')
        result, output = self.check()
        self.assertEqual(result, 1)
        self.assertIn('no ARM64 Linux ELF files found', output)

    def test_abi_boundaries(self):
        for version, expected in (
            ('GLIBC_2.41', 0),
            ('GLIBC_2.42', 1),
            ('GLIBCXX_3.4.30', 0),
            ('GLIBCXX_3.4.31', 1),
            ('CXXABI_1.3.13', 0),
            ('CXXABI_1.3.14', 1),
        ):
            with self.subTest(version=version):
                path = self.elf('bin/geniex-bench', [version])
                result, _ = self.check(path)
                self.assertEqual(result, expected)

    def test_readelf_failure_fails_audit(self):
        self.elf('bin/geniex-bench')
        result, output = self.check(readelf=subprocess.CalledProcessError(1, ['readelf']))
        self.assertEqual(result, 1)
        self.assertIn('cannot inspect Linux ABI', output)


if __name__ == '__main__':
    unittest.main()
