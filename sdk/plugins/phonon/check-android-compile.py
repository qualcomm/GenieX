# SPDX-License-Identifier: BSD-3-Clause
"""Cross-compile shared Phonon/GenieX sources without loading an Android model.

This checks compilation only, not linking, APK packaging or NPU execution.
"""

import argparse
import json
import os
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ndk", type=Path, required=True)
    parser.add_argument("--qairt", type=Path, required=True)
    parser.add_argument("--phonon-source", type=Path, required=True)
    parser.add_argument("--json-include", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--api", type=int, default=31)
    args = parser.parse_args()
    if args.api < 31:
        parser.error("Phonon Android targets Android 12/API 31 or newer")
    repo = Path(__file__).resolve().parents[3]
    engine = args.phonon_source.resolve() / "qnn/runner"
    prebuilt = args.ndk.resolve() / "toolchains/llvm/prebuilt"
    compiler_name = "clang++.exe" if os.name == "nt" else "clang++"
    compilers = list(prebuilt.glob(f"*/bin/{compiler_name}"))
    if len(compilers) != 1:
        parser.error("Expected one host NDK clang++ executable")
    compiler = compilers[0]
    includes = [
        repo / "sdk/include",
        repo / "bindings/android/app/src/main/cpp",
        engine,
        args.json_include.resolve(),
        args.qairt.resolve() / "include/QNN",
    ]
    for path in [
        engine / "phonon.hpp",
        args.json_include / "nlohmann/json.hpp",
        args.qairt / "include/QNN/QnnInterface.h",
    ]:
        if not path.is_file():
            parser.error(f"Missing dependency: {path}")
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    config = args.out / "compile-config.h"
    config.write_text('#define PROJECT_SOURCE_DIR "geniex"\n', encoding="ascii")
    sources = [
        repo / "sdk/plugins/phonon/plugin.cpp",
        engine / "ptts_text.cpp",
        engine / "qnn_model.cpp",
        engine / "qnn_log.cpp",
        engine / "phonon.cpp",
        repo / "bindings/android/app/src/main/cpp/tts_bridge_jni.cpp",
    ]
    commands = []
    for source in sources:
        output = args.out / f"{source.stem}.o"
        command = [
            str(compiler),
            f"--target=aarch64-linux-android{args.api}",
            "-std=c++17",
            "-fPIC",
            "-DGENIEX_SHARED",
            "-include",
            str(config),
        ]
        command += [f"-I{path}" for path in includes]
        command += ["-c", str(source), "-o", str(output)]
        subprocess.run(command, check=True)
        commands.append(dict(directory=str(repo), file=str(source), arguments=command))
        print(f"Compiled {source.name} for Android ARM64", flush=True)
    (args.out / "compile_commands.json").write_text(
        json.dumps(commands, indent=2), encoding="utf-8"
    )
    print(
        "Compile checks passed; Android linking, packaging and hardware tests remain required."
    )


if __name__ == "__main__":
    main()
