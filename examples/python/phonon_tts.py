# SPDX-License-Identifier: BSD-3-Clause
"""Stream Phonon PCM through the GenieX C ABI and write measured WAV/JSON files."""

import argparse
import array
import json
import os
import sys
import time
import wave
from dataclasses import asdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", required=True, help="Compiled Phonon QNN bundle directory"
    )
    parser.add_argument(
        "--runtime", required=True, help="QAIRT host runtime library directory"
    )
    parser.add_argument("--text-library", required=True)
    parser.add_argument("--soc", required=True)
    parser.add_argument("--language", default="en")
    parser.add_argument("--voice", default="")
    parser.add_argument("--device", default="npu")
    parser.add_argument(
        "--text",
        default="Hello from Phonon, running locally through GenieX on the Snapdragon NPU.",
    )
    parser.add_argument("--repeats", type=int, default=6)
    parser.add_argument(
        "--dependency-dir",
        action="append",
        default=[],
        help="Additional trusted build-tree DLL directory",
    )
    parser.add_argument("--out", type=Path, default=Path("phonon-results"))
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")

    # Retain Windows DLL directory handles until after model destruction. These
    # are explicitly trusted runtime locations, never model download directories.
    dll_handles = []
    if sys.platform == "win32":
        paths = [args.runtime, str(Path(args.text_library).resolve().parent)]
        paths.extend(args.dependency_dir)
        lib = os.environ.get("GENIEX_LIB_PATH")
        if lib:
            paths.append(lib if Path(lib).is_dir() else str(Path(lib).parent))
        dll_handles = [os.add_dll_directory(str(Path(p).resolve())) for p in paths]

    from geniex import GenieXTTS, set_qairt_runtime_path

    set_qairt_runtime_path(str(Path(args.runtime).resolve()))

    args.out.mkdir(parents=True, exist_ok=True)
    load_start = time.perf_counter()
    records = []
    try:
        with GenieXTTS(
            args.model,
            runtime_path=args.runtime,
            text_library=args.text_library,
            soc_model=args.soc,
            language=args.language,
            device=args.device,
            verify_npu=True,
        ) as model:
            load_ms = (time.perf_counter() - load_start) * 1000
            for index in range(args.repeats):
                start, first_callback, frames = time.perf_counter(), None, 0

                def audio(_data, _rate, _start=start):
                    nonlocal first_callback, frames
                    frames += 1
                    if first_callback is None:
                        first_callback = (time.perf_counter() - _start) * 1000
                    return True

                result = model.synthesize(args.text, args.voice, audio)
                assert (
                    result.samples > 0
                    and result.sample_rate == 24000
                    and not result.cancelled
                )
                assert result.prefill_npu_ms > 0 and result.step_npu_ms > 0
                floats = array.array("f")
                floats.frombytes(result.pcm)
                assert (
                    len(floats) == result.samples
                    and max(abs(v) for v in floats) > 0.001
                )
                pcm16 = array.array(
                    "h", (max(-32768, min(32767, round(v * 32767))) for v in floats)
                )
                if sys.byteorder != "little":
                    pcm16.byteswap()
                wav_path = args.out / f"run-{index}.wav"
                with wave.open(str(wav_path), "wb") as wav:
                    wav.setnchannels(1)
                    wav.setsampwidth(2)
                    wav.setframerate(result.sample_rate)
                    wav.writeframes(pcm16.tobytes())
                record = asdict(result)
                record.pop("pcm")
                record.update(
                    run=index,
                    warm=index > 0,
                    model_load_ms=load_ms,
                    measured_callback_ms=first_callback,
                    callbacks=frames,
                    rtf=result.rtf,
                    audio_seconds=result.samples / result.sample_rate,
                    wav=str(wav_path.resolve()),
                )
                records.append(record)
                print(json.dumps(record), flush=True)
    finally:
        for handle in dll_handles:
            handle.close()
    (args.out / "results.json").write_text(
        json.dumps(records, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
