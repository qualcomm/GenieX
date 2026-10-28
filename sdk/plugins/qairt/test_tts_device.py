# SPDX-License-Identifier: BSD-3-Clause
"""Hardware regression checks. No model downloads, CPU or GPU execution.

See README.md. Run in a fresh process with profiling and trusted DLL paths set.
"""

import argparse
import ctypes
import json
import os
from pathlib import Path


def loaded_qnn_libraries():
    """Inspect this process, not a launcher or a different installed SDK."""
    from ctypes import wintypes

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.EnumProcessModulesEx.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(wintypes.HMODULE),
        wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
        wintypes.DWORD,
    ]
    psapi.GetModuleFileNameExW.argtypes = [
        wintypes.HANDLE,
        wintypes.HMODULE,
        wintypes.LPWSTR,
        wintypes.DWORD,
    ]
    modules = (wintypes.HMODULE * 1024)()
    needed = wintypes.DWORD()
    process = kernel.GetCurrentProcess()
    assert psapi.EnumProcessModulesEx(
        process, modules, ctypes.sizeof(modules), ctypes.byref(needed), 3
    )
    assert needed.value <= ctypes.sizeof(modules)
    result = {}
    for module in modules[: needed.value // ctypes.sizeof(wintypes.HMODULE)]:
        name = ctypes.create_unicode_buffer(32768)
        assert psapi.GetModuleFileNameExW(process, module, name, len(name))
        path = Path(name.value)
        if path.name.lower().startswith("qnn"):
            result[path.name.lower()] = str(path.resolve())
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--runtime", required=True)
    parser.add_argument("--text-library", required=True)
    parser.add_argument("--soc", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dependency-dir", action="append", default=[])
    args = parser.parse_args()
    dll_handles = []
    if os.name == "nt":
        dirs = [
            Path(args.runtime),
            Path(args.text_library).resolve().parent,
            Path(os.environ["GENIEX_LIB_PATH"]).resolve().parent,
        ]
        dirs.extend(Path(p) for p in args.dependency_dir)
        dll_handles = [os.add_dll_directory(str(p.resolve())) for p in dirs]

    from geniex import (
        GenieXError,
        GenieXTTS,
        get_compute_unit_list,
        get_qairt_runtime_path,
        init,
        set_qairt_runtime_path,
    )

    set_qairt_runtime_path(str(Path(args.runtime).resolve()))

    records = []
    init()
    try:
        set_qairt_runtime_path(str(Path(args.runtime).resolve()))
        raise AssertionError("runtime selection changed after init")
    except GenieXError as exc:
        assert exc.code == -100008
        records.append(dict(test="runtime_locked_after_init", passed=True))

    def expect_failure(label, bundle, **overrides):
        params = dict(
            runtime_path=args.runtime,
            text_library=args.text_library,
            soc_model=args.soc,
            verify_npu=True,
        )
        params.update(overrides)
        try:
            with GenieXTTS(bundle, **params):
                raise AssertionError(f"{label}: unexpectedly loaded")
        except GenieXError as exc:
            expected = {
                "reject_cpu": -100002,
                "reject_gpu": -100002,
                "wrong_soc": -100201,
                "conflicting_runtime_assertion": -100001,
                "missing_text_frontend": -100201,
            }
            assert exc.code == expected[label], (label, exc.code)
            records.append(
                dict(test=label, bundle=str(bundle), status=exc.code, passed=True)
            )

    try:
        for bundle in args.models:
            expect_failure("reject_cpu", bundle, device="cpu")
            expect_failure("reject_gpu", bundle, device="gpu")
            expect_failure("wrong_soc", bundle, soc_model="NOT_A_SUPPORTED_SOC")
            expect_failure(
                "conflicting_runtime_assertion",
                bundle,
                runtime_path=str(Path(bundle).resolve() / "no-runtime"),
            )
            expect_failure(
                "missing_text_frontend",
                bundle,
                text_library=str(Path(bundle).resolve() / "no-text.dll"),
            )
            with GenieXTTS(
                bundle,
                text_library=args.text_library,
                soc_model=args.soc,
                verify_npu=True,
            ) as model:
                if os.name == "nt":
                    libraries = loaded_qnn_libraries()
                    assert "qnnhtp.dll" in libraries and "qnnsystem.dll" in libraries
                    assert (
                        "qnncpu.dll" not in libraries and "qnngpu.dll" not in libraries
                    )
                    for name in ("qnnhtp.dll", "qnnsystem.dll"):
                        assert Path(libraries[name]).parent.samefile(args.runtime)
                    records.append(
                        dict(
                            test="shared_htp_runtime_loaded",
                            bundle=bundle,
                            libraries=libraries,
                            passed=True,
                        )
                    )
                assert [device for device, _ in get_compute_unit_list("qairt")] == [
                    "NPU"
                ]
                records.append(
                    dict(test="only_npu_advertised", bundle=bundle, passed=True)
                )
                assert get_qairt_runtime_path() == str(Path(args.runtime).resolve())
                records.append(
                    dict(test="inherits_process_runtime", bundle=bundle, passed=True)
                )

                def checked(text, **kwargs):
                    result = model.synthesize(text, **kwargs)
                    assert result.samples > 0 and len(result.pcm) == result.samples * 4
                    if not result.cancelled:
                        assert result.prefill_npu_ms > 0 and result.step_npu_ms > 0
                    return result

                assert checked(
                    "Cancel this speech at the first audio frame.",
                    on_audio=lambda *_: False,
                ).cancelled
                records.append(dict(test="callback_cancel", bundle=bundle, passed=True))
                assert checked(
                    "Cancel explicitly from the callback.",
                    on_audio=lambda *_: model.cancel(),
                ).cancelled
                records.append(dict(test="explicit_cancel", bundle=bundle, passed=True))
                try:
                    model.synthesize(
                        "Test a callback exception.", on_audio=lambda *_: 1 / 0
                    )
                    raise AssertionError("exception not propagated")
                except ZeroDivisionError:
                    records.append(
                        dict(test="callback_exception", bundle=bundle, passed=True)
                    )
                result = checked("The model is still reusable after cancellation.")
                assert (
                    result.prefill_calls == 1
                )  # Counts must not accumulate over prior requests.
                records.append(
                    dict(
                        test="reuse_and_profile_deltas",
                        bundle=bundle,
                        passed=True,
                        prefill_calls=result.prefill_calls,
                        step_calls=result.step_calls,
                    )
                )
                text = (
                    "Welcome to your local voice assistant. This longer passage checks that speech continues "
                    "across several text chunks, while every neural graph runs on the Hexagon accelerator. "
                    "You can listen to documents, review your work, or ask the assistant for help. "
                    "No cloud speech service is needed for this example. "
                )
                result = checked(text)
                assert result.prefill_calls > 1
                records.append(
                    dict(
                        test="long_multichunk",
                        bundle=bundle,
                        passed=True,
                        samples=result.samples,
                        prefill_calls=result.prefill_calls,
                        step_calls=result.step_calls,
                        prefill_npu_ms=result.prefill_npu_ms,
                        step_npu_ms=result.step_npu_ms,
                    )
                )
                try:
                    model.synthesize(
                        "Unknown voice test.", voice="voice_does_not_exist"
                    )
                    raise AssertionError("unknown voice accepted")
                except GenieXError as exc:
                    assert exc.code == -100001
                assert not checked(
                    "An invalid voice does not poison the model."
                ).cancelled
                records.append(
                    dict(test="invalid_voice_then_reuse", bundle=bundle, passed=True)
                )
    finally:
        for handle in dll_handles:
            handle.close()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(json.dumps(dict(passed=len(records), results=str(args.out.resolve()))))


if __name__ == "__main__":
    main()
