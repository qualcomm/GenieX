# SPDX-License-Identifier: BSD-3-Clause
"""Hardware integration: resident Qwen -> Phonon -> Qwen, strict QNN NPU.

No downloads or alternate backends. Launch in a fresh process; select bundles
compiled for one compatible runtime. Scheduling here belongs to the application,
not to a new SDK-wide scheduler. See README.md for required loader paths.
"""

import argparse
import array
import ctypes
import json
import logging
import os
import threading
import time
import wave
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path


def qnn_modules():
    """Record actual loaded host modules on Windows, not just environment paths."""
    if os.name != "nt":
        return None
    from ctypes import wintypes

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.EnumProcessModules.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(wintypes.HMODULE),
        wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
    ]
    psapi.GetModuleFileNameExW.argtypes = [
        wintypes.HANDLE,
        wintypes.HMODULE,
        wintypes.LPWSTR,
        wintypes.DWORD,
    ]
    process = kernel.GetCurrentProcess()
    handles = (wintypes.HMODULE * 4096)()
    needed = wintypes.DWORD()
    if not psapi.EnumProcessModules(
        process, handles, ctypes.sizeof(handles), ctypes.byref(needed)
    ):
        raise ctypes.WinError(ctypes.get_last_error())
    assert needed.value <= ctypes.sizeof(handles), "module inventory truncated"
    paths = []
    for handle in handles[: needed.value // ctypes.sizeof(wintypes.HMODULE)]:
        name = ctypes.create_unicode_buffer(32768)
        if not psapi.GetModuleFileNameExW(process, handle, name, len(name)):
            raise ctypes.WinError(ctypes.get_last_error())
        path = Path(name.value)
        if path.name.lower() in {
            "qnnhtp.dll",
            "qnnsystem.dll",
            "qnncpu.dll",
            "qnngpu.dll",
        }:
            paths.append(str(path))
    return sorted(paths)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qwen", type=Path, required=True)
    parser.add_argument("--models", nargs="+", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--text-library", type=Path, required=True)
    parser.add_argument("--soc", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    for path in [args.qwen, args.runtime, *args.models]:
        if not path.is_dir():
            parser.error(f"Required local directory not found: {path}")
    if not args.text_library.is_file():
        parser.error(f"Trusted text frontend not found: {args.text_library}")
    args.out.mkdir(parents=True, exist_ok=True)
    os.environ["PHONON_QNN_PROFILE"] = "1"
    host = args.runtime.resolve()
    if (host / "lib").is_dir():
        host = host / "lib/aarch64-windows-msvc" if os.name == "nt" else host
    dll_handles = []
    if os.name == "nt":
        dll_handles = [
            os.add_dll_directory(str(path))
            for path in (host, args.text_library.resolve().parent)
        ]

    logging.basicConfig(level=logging.DEBUG)
    from geniex import (
        AutoModelForCausalLM,
        GenieXTTS,
        init,
        set_log_level,
        set_qairt_runtime_path,
    )

    set_qairt_runtime_path(str(args.runtime.resolve()))
    init()
    set_log_level("debug")
    records = []
    schedule = threading.Lock()
    llm = AutoModelForCausalLM.from_pretrained(
        str(args.qwen.resolve()), device_map="qairt:npu"
    )

    def ask(expected):
        with schedule:
            llm.reset()
            prompt = llm.tokenizer.apply_chat_template(
                [{"role": "user", "content": f"Reply with exactly: {expected}"}],
                enable_thinking=False,
                add_generation_prompt=True,
            )
            start = time.perf_counter()
            result = llm.generate(prompt, max_new_tokens=24)
            assert expected.lower() in result.text.lower(), repr(result.text)
            assert result.profile.backend == "qairt", asdict(result.profile)
            assert (result.profile.device or "").lower() == "npu", asdict(
                result.profile
            )
            records.append(
                dict(
                    test="qwen",
                    expected=expected,
                    text=result.text,
                    wall_ms=1000 * (time.perf_counter() - start),
                    profile=asdict(result.profile),
                )
            )
            return result.text

    def check_speech(result, name):
        assert (
            not result.cancelled
            and result.prefill_npu_ms > 0
            and result.step_npu_ms > 0
        )
        pcm = array.array("f", result.pcm)
        assert len(pcm) == result.samples and max(abs(x) for x in pcm) > 0.001
        with wave.open(str(args.out / f"{name}.wav"), "wb") as wav:
            wav.setparams((1, 2, result.sample_rate, 0, "NONE", "not compressed"))
            wav.writeframes(
                array.array(
                    "h", (round(max(-1, min(1, x)) * 32767) for x in pcm)
                ).tobytes()
            )
        records.append(
            dict(
                test=name,
                first_audio_ms=result.first_audio_ms,
                rtf=result.rtf,
                prefill_npu_ms=result.prefill_npu_ms,
                step_npu_ms=result.step_npu_ms,
            )
        )

    try:
        ask("NPU READY")
        for index, bundle in enumerate(args.models):
            with GenieXTTS(
                str(bundle.resolve()),
                text_library=str(args.text_library.resolve()),
                soc_model=args.soc,
                verify_npu=True,
            ) as speech:
                for repeat in range(3):
                    with schedule:
                        result = speech.synthesize(
                            "Welcome. Local speech and language share one selected NPU runtime."
                        )
                    check_speech(result, f"model-{index}-speech-{repeat}")
                with schedule:
                    cancelled = speech.synthesize(
                        "Cancel this sentence.", on_audio=lambda *_: False
                    )
                assert cancelled.cancelled
                records.append(dict(test="cancel", model=index, passed=True))
                ask("NPU STILL READY")

                # An overlapping application request must queue, not execute two
                # engines concurrently. Hold a live speech callback while asking.
                entered, release, requested = (
                    threading.Event(),
                    threading.Event(),
                    threading.Event(),
                )

                def gate(*_):
                    entered.set()
                    assert release.wait(10), "test gate timed out"

                def gated_speech():
                    with schedule:
                        return speech.synthesize(
                            "The application queues work safely.", on_audio=gate
                        )

                def queued_question():
                    requested.set()
                    return ask("QUEUE READY")

                with ThreadPoolExecutor(max_workers=2) as pool:
                    first = pool.submit(gated_speech)
                    try:
                        assert entered.wait(10), "no speech callback"
                        second = pool.submit(queued_question)
                        assert requested.wait(10)
                        assert (
                            not second.done()
                        ), "language request bypassed application scheduling"
                    finally:
                        release.set()
                    check_speech(
                        first.result(timeout=60), f"model-{index}-queued-speech"
                    )
                    second.result(timeout=60)
                records.append(
                    dict(test="serialized_overlap", model=index, passed=True)
                )
        modules = qnn_modules()
        if modules is not None:
            names = {Path(path).name.lower() for path in modules}
            assert {"qnnhtp.dll", "qnnsystem.dll"}.issubset(names), modules
            assert not {"qnncpu.dll", "qnngpu.dll"} & names, modules
            assert all(Path(path).parent.samefile(host) for path in modules), modules
        report = dict(
            passed=True,
            records=records,
            loaded_qnn_modules=modules,
            scheduling="application lock; no SDK-wide concurrent-engine guarantee",
        )
        (args.out / "results.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        print(
            json.dumps(
                dict(passed=True, checks=len(records), results=str(args.out.resolve()))
            )
        )
    finally:
        llm.close()
        for handle in dll_handles:
            handle.close()


if __name__ == "__main__":
    main()
